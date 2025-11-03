import asyncio
import time
from pymilvus import connections, Collection, utility


class MilvusSyncService:
    def __init__(self):
        # Source (remote) and destination (local)
        self.source_host = "103.252.0.129"
        self.source_port = "19530"
        self.dest_host = "localhost"
        self.dest_port = "19530"

        # Collections to sync
        self.collections = ["document_embeddings", "faq_embeddings"]

        self.batch_size = 100
        self.sync_interval = 10  # seconds between sync checks

    async def connect_all(self):
        """Connect both Milvus servers"""
        try:
            connections.connect("source", host=self.source_host, port=self.source_port)
            connections.connect("destination", host=self.dest_host, port=self.dest_port)
            print(f"✓ Connected to both source and destination Milvus")
        except Exception as e:
            print(f"✗ Connection error: {e}")

    async def get_all_ids(self, collection_name: str, alias: str):
        """Get all IDs in a collection"""
        try:
            if not utility.has_collection(collection_name, using=alias):
                print(f"Collection {collection_name} not found in {alias}")
                return set()

            collection = Collection(collection_name, using=alias)
            collection.load()
            schema = collection.schema
            id_field = schema.primary_field.name

            ids = set()
            offset = 0
            limit = self.batch_size
            total = collection.num_entities

            while offset < total:
                results = collection.query(
                    expr="",
                    output_fields=[id_field],
                    limit=limit,
                    offset=offset
                )
                if not results:
                    break
                ids.update(r[id_field] for r in results)
                offset += len(results)
            return ids
        except Exception as e:
            print(f"✗ Error getting IDs for {collection_name} ({alias}): {e}")
            return set()

    async def fetch_records_by_ids(self, collection_name: str, alias: str, ids):
        """Fetch records by given ID list"""
        try:
            if not ids:
                return []

            collection = Collection(collection_name, using=alias)
            schema = collection.schema
            fields = [f.name for f in schema.fields]
            id_field = schema.primary_field.name

            # Batch query if many IDs
            results = []
            id_list = list(ids)
            for i in range(0, len(id_list), self.batch_size):
                batch_ids = id_list[i:i + self.batch_size]
                expr = f"{id_field} in {batch_ids}"
                data = collection.query(expr=expr, output_fields=fields)
                results.extend(data)

            return results
        except Exception as e:
            print(f"✗ Error fetching records: {e}")
            return []

    async def insert_new_records(self, collection_name: str, data):
        """Insert new records into local collection"""
        if not data:
            return
        try:
            collection = Collection(collection_name, using="destination")
            field_names = list(data[0].keys())
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                entities = [[record[field] for record in batch] for field in field_names]
                collection.insert(entities)
            collection.load()
            print(f"✓ Inserted {len(data)} new records into {collection_name}")
        except Exception as e:
            print(f"✗ Insert error: {e}")

    async def sync_collection(self, collection_name: str):
        """Compare source vs destination and sync new records"""
        print(f"\n🔍 Checking for updates in {collection_name}...")
        source_ids = await self.get_all_ids(collection_name, "source")
        dest_ids = await self.get_all_ids(collection_name, "destination")

        new_ids = source_ids - dest_ids
        if not new_ids:
            print(f"→ No new IDs in {collection_name}")
            return

        print(f"→ Found {len(new_ids)} new records to sync for {collection_name}")
        new_records = await self.fetch_records_by_ids(collection_name, "source", new_ids)
        await self.insert_new_records(collection_name, new_records)

    async def run_service(self):
        """Run continuous sync loop"""
        await self.connect_all()
        print("🚀 Starting Milvus Incremental Sync Service...")

        while True:
            for col in self.collections:
                await self.sync_collection(col)
            print(f"⏳ Sleeping {self.sync_interval} seconds before next check...\n")
            await asyncio.sleep(self.sync_interval)


if __name__ == "__main__":
    asyncio.run(MilvusSyncService().run_service())
