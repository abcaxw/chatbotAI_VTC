"""
Milvus Data Migration Script
Migrate all data from remote Milvus to local Milvus
"""

from pymilvus import connections, Collection, utility
import asyncio
from typing import List, Dict, Any
import json


class MilvusMigration:
    def __init__(self):
        # Source (remote) connection
        self.source_host = "103.252.0.129"
        self.source_port = "19530"

        # Destination (local) connection
        self.dest_host = "localhost"
        self.dest_port = "19530"

        # Collection names
        self.doc_collection_name = "document_embeddings"
        self.faq_collection_name = "faq_embeddings"

        self.batch_size = 100

    async def connect_source(self):
        """Connect to source Milvus"""
        try:
            connections.connect(
                alias="source",
                host=self.source_host,
                port=self.source_port
            )
            print(f"✓ Connected to source Milvus at {self.source_host}:{self.source_port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to source: {e}")
            return False

    async def connect_destination(self):
        """Connect to destination Milvus"""
        try:
            connections.connect(
                alias="destination",
                host=self.dest_host,
                port=self.dest_port
            )
            print(f"✓ Connected to destination Milvus at {self.dest_host}:{self.dest_port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to destination: {e}")
            return False

    async def export_collection_data(self, collection_name: str, alias: str) -> List[Dict]:
        """Export all data from a collection"""
        try:
            if not utility.has_collection(collection_name, using=alias):
                print(f"Collection {collection_name} does not exist in {alias}")
                return []

            collection = Collection(collection_name, using=alias)
            collection.load()

            # Get total count
            total_count = collection.num_entities
            print(f"Exporting {total_count} entities from {collection_name}...")

            if total_count == 0:
                return []

            # Get all data using query (query all)
            all_data = []

            # Query in batches
            offset = 0
            limit = self.batch_size

            # Get schema to know field names
            schema = collection.schema
            field_names = [field.name for field in schema.fields if not field.is_primary]
            field_names.insert(0, schema.primary_field.name)  # Add primary key first

            while offset < total_count:
                try:
                    # Query batch
                    results = collection.query(
                        expr="",  # Empty expr means query all
                        output_fields=field_names,
                        limit=limit,
                        offset=offset
                    )

                    if not results:
                        break

                    all_data.extend(results)
                    offset += len(results)
                    print(f"  Exported {offset}/{total_count} entities...")

                except Exception as e:
                    print(f"Error querying batch at offset {offset}: {e}")
                    break

            print(f"✓ Exported {len(all_data)} entities from {collection_name}")
            return all_data

        except Exception as e:
            print(f"✗ Export error for {collection_name}: {e}")
            return []

    async def import_collection_data(self, collection_name: str, data: List[Dict], alias: str) -> bool:
        """Import data into a collection, skipping existing IDs"""
        try:
            if not data:
                print(f"No data to import for {collection_name}")
                return True

            if not utility.has_collection(collection_name, using=alias):
                print(f"✗ Collection {collection_name} does not exist in {alias}")
                return False

            collection = Collection(collection_name, using=alias)
            collection.load()

            # Get primary key field name
            schema = collection.schema
            primary_field = schema.primary_field.name

            # Fetch all existing IDs in destination
            print(f"Fetching existing IDs from {collection_name}...")
            existing_ids = set()
            offset = 0
            limit = self.batch_size

            total_count = collection.num_entities
            while offset < total_count:
                try:
                    results = collection.query(
                        expr="",
                        output_fields=[primary_field],
                        limit=limit,
                        offset=offset
                    )
                    if not results:
                        break
                    existing_ids.update(r[primary_field] for r in results)
                    offset += len(results)
                except Exception as e:
                    print(f"✗ Error fetching existing IDs batch at offset {offset}: {e}")
                    break

            print(f"Found {len(existing_ids)} existing IDs in destination.")

            # Filter out records that already exist
            new_data = [record for record in data if record[primary_field] not in existing_ids]
            print(f"{len(new_data)} new entities to import out of {len(data)} total.")

            if not new_data:
                print(f"No new data to import for {collection_name}")
                return True

            field_names = list(new_data[0].keys())
            total_inserted = 0

            for i in range(0, len(new_data), self.batch_size):
                batch = new_data[i:i + self.batch_size]

                # Organize data by fields
                entities = []
                for field_name in field_names:
                    field_data = [record[field_name] for record in batch]
                    entities.append(field_data)

                try:
                    collection.insert(entities)
                    total_inserted += len(batch)
                    print(f"  Imported {total_inserted}/{len(new_data)} new entities...")
                except Exception as e:
                    print(f"✗ Error importing batch {i // self.batch_size + 1}: {e}")
                    continue

            collection.load()
            print(f"✓ Imported {total_inserted} new entities into {collection_name}")
            return True

        except Exception as e:
            print(f"✗ Import error for {collection_name}: {e}")
            return False

    async def migrate_collection(self, collection_name: str):
        """Migrate a specific collection"""
        print(f"\n{'=' * 60}")
        print(f"Migrating collection: {collection_name}")
        print(f"{'=' * 60}")

        # Export from source
        data = await self.export_collection_data(collection_name, "source")

        if not data:
            print(f"No data to migrate for {collection_name}")
            return False

        # Import to destination
        success = await self.import_collection_data(collection_name, data, "destination")

        return success

    async def verify_migration(self, collection_name: str):
        """Verify data migration"""
        try:
            # Check source
            if utility.has_collection(collection_name, using="source"):
                source_col = Collection(collection_name, using="source")
                source_col.load()
                source_count = source_col.num_entities
            else:
                source_count = 0

            # Check destination
            if utility.has_collection(collection_name, using="destination"):
                dest_col = Collection(collection_name, using="destination")
                dest_col.load()
                dest_count = dest_col.num_entities
            else:
                dest_count = 0

            print(f"\nVerification for {collection_name}:")
            print(f"  Source count: {source_count}")
            print(f"  Destination count: {dest_count}")

            if source_count == dest_count:
                print(f"  ✓ Migration successful!")
                return True
            else:
                print(f"  ✗ Count mismatch!")
                return False

        except Exception as e:
            print(f"✗ Verification error: {e}")
            return False

    async def run_migration(self):
        """Run full migration process"""
        print("=" * 60)
        print("Milvus Data Migration Tool")
        print("=" * 60)
        print(f"Source: {self.source_host}:{self.source_port}")
        print(f"Destination: {self.dest_host}:{self.dest_port}")
        print("=" * 60)

        # Connect to both databases
        source_ok = await self.connect_source()
        dest_ok = await self.connect_destination()

        if not (source_ok and dest_ok):
            print("\n✗ Failed to establish connections. Aborting migration.")
            return False

        # Migrate document_embeddings collection
        await self.migrate_collection(self.doc_collection_name)
        await self.verify_migration(self.doc_collection_name)

        # Migrate faq_embeddings collection
        await self.migrate_collection(self.faq_collection_name)
        await self.verify_migration(self.faq_collection_name)

        # Disconnect
        connections.disconnect("source")
        connections.disconnect("destination")

        print("\n" + "=" * 60)
        print("Migration completed!")
        print("=" * 60)


async def main():
    """Main function"""
    migrator = MilvusMigration()
    await migrator.run_migration()


if __name__ == "__main__":
    asyncio.run(main())