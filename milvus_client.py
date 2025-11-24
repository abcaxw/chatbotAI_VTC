from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict, Any
import asyncio
import uuid


class MilvusManager:
    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.collection_name = "document_embeddings"
        self.faq_collection_name = "faq_embeddings"
        self.collection = None
        self.faq_collection = None

        # THAY ĐỔI: 1024 -> 768 dimensions
        self.embedding_dim = 768

        # Field length limits
        self.max_id_length = 190
        self.max_document_id_length = 90
        self.max_description_length = 60000
        self.max_question_length = 60000
        self.max_answer_length = 60000

    async def initialize(self):
        """Initialize Milvus connection and create collections"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"Connected to Milvus at {self.host}:{self.port}")

            await self.create_collection()
            await self.create_faq_collection()

        except Exception as e:
            print(f"Milvus initialization error: {e}")
            raise e

    def _validate_and_truncate(self, data: Dict[str, Any], field_limits: Dict[str, int]) -> Dict[str, Any]:
        """Validate and truncate fields to fit Milvus limits"""
        validated = data.copy()

        for field, max_length in field_limits.items():
            if field in validated and isinstance(validated[field], str):
                if len(validated[field]) > max_length:
                    print(f"Warning: Truncating {field} from {len(validated[field])} to {max_length} chars")
                    validated[field] = validated[field][:max_length - 3] + "..."

        return validated

    async def create_collection(self):
        """Create collection with 768D vectors"""
        try:
            if utility.has_collection(self.collection_name):
                print(f"Collection {self.collection_name} already exists")
                self.collection = Collection(self.collection_name)
                return

            # THAY ĐỔI: dim=768
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65000),
                FieldSchema(name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Document embeddings collection (768D)"
            )

            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )

            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 768}
            }

            self.collection.create_index(
                field_name="description_vector",
                index_params=index_params
            )

            print(f"Collection {self.collection_name} created successfully with 768D vectors")

        except Exception as e:
            print(f"Collection creation error: {e}")
            raise e

    async def create_faq_collection(self):
        """Create FAQ collection with 768D vectors"""
        try:
            if utility.has_collection(self.faq_collection_name):
                print(f"Collection {self.faq_collection_name} already exists")
                self.faq_collection = Collection(self.faq_collection_name)
                return

            # THAY ĐỔI: dim=768
            fields = [
                FieldSchema(name="faq_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=65000),
                FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65000),
                FieldSchema(name="question_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
            ]

            schema = CollectionSchema(
                fields=fields,
                description="FAQ embeddings collection (768D)"
            )

            self.faq_collection = Collection(
                name=self.faq_collection_name,
                schema=schema,
                using='default'
            )

            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 768}
            }

            self.faq_collection.create_index(
                field_name="question_vector",
                index_params=index_params
            )

            print(f"Collection {self.faq_collection_name} created successfully with 768D vectors")

        except Exception as e:
            print(f"FAQ Collection creation error: {e}")
            raise e

    async def insert_embeddings(self, embeddings_data: List[Dict]) -> int:
        """Insert embeddings into collection with validation"""
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            if not embeddings_data:
                return 0

            field_limits = {
                "id": self.max_id_length,
                "document_id": self.max_document_id_length,
                "description": self.max_description_length
            }

            validated_data = []
            for item in embeddings_data:
                if not all(key in item for key in ["id", "document_id", "description", "description_vector"]):
                    print(f"Skipping item missing required fields: {item.keys()}")
                    continue

                validated_item = self._validate_and_truncate(item, field_limits)

                # Validate vector dimension = 768
                if len(validated_item["description_vector"]) != self.embedding_dim:
                    print(f"Skipping item with incorrect vector dimension: {len(validated_item['description_vector'])}")
                    continue

                validated_data.append(validated_item)

            if not validated_data:
                print("No valid data to insert")
                return 0

            # Prepare data for insertion
            ids = [item["id"] for item in validated_data]
            document_ids = [item["document_id"] for item in validated_data]
            descriptions = [item["description"] for item in validated_data]
            vectors = [item["description_vector"] for item in validated_data]

            # Insert in batches
            batch_size = 100
            total_inserted = 0

            for i in range(0, len(validated_data), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_document_ids = document_ids[i:i + batch_size]
                batch_descriptions = descriptions[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]

                entities = [batch_ids, batch_document_ids, batch_descriptions, batch_vectors]

                try:
                    insert_result = self.collection.insert(entities)
                    total_inserted += len(batch_ids)
                    print(f"Inserted batch {i // batch_size + 1}: {len(batch_ids)} items")
                except Exception as batch_error:
                    print(f"Error inserting batch {i // batch_size + 1}: {batch_error}")
                    continue

            self.collection.load()
            print(f"Total inserted: {total_inserted} embeddings")
            return total_inserted

        except Exception as e:
            print(f"Insert error: {e}")
            raise e

    async def insert_faq(self, faq_id: str, question: str, answer: str, question_vector: List[float]) -> bool:
        """Insert FAQ with 768D vector"""
        try:
            if not self.faq_collection:
                raise Exception("FAQ Collection not initialized")

            if len(faq_id) > 90:
                faq_id = faq_id[:90]
            if len(question) > self.max_question_length:
                question = question[:self.max_question_length - 3] + "..."
            if len(answer) > self.max_answer_length:
                answer = answer[:self.max_answer_length - 3] + "..."

            # Validate 768D
            if len(question_vector) != self.embedding_dim:
                print(f"Invalid vector dimension: {len(question_vector)}")
                return False

            entities = [[faq_id], [question], [answer], [question_vector]]
            insert_result = self.faq_collection.insert(entities)
            self.faq_collection.load()

            print(f"Inserted FAQ with id: {faq_id}")
            return True

        except Exception as e:
            print(f"FAQ Insert error: {e}")
            return False

    async def delete_faq(self, faq_id: str) -> bool:
        """Delete FAQ by ID"""
        try:
            if not self.faq_collection:
                raise Exception("FAQ Collection not initialized")

            expr = f'faq_id == "{faq_id}"'
            delete_result = self.faq_collection.delete(expr)

            print(f"Deleted FAQ with id: {faq_id}")
            return True

        except Exception as e:
            print(f"FAQ Delete error: {e}")
            return False

    async def delete_document(self, document_id: str) -> int:
        """Delete all embeddings for a document"""
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            expr = f'document_id == "{document_id}"'
            delete_result = self.collection.delete(expr)

            print(f"Deleted all embeddings for document_id: {document_id}")
            return True

        except Exception as e:
            print(f"Document Delete error: {e}")
            return False

    async def search_similar(self, query_vector: List[float], limit: int = 10, min_score: float = 0.0) -> List[Dict]:
        """Search for similar embeddings"""
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            if len(query_vector) != self.embedding_dim:
                raise Exception(f"Query vector dimension mismatch: {len(query_vector)} != {self.embedding_dim}")

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            results = self.collection.search(
                data=[query_vector],
                anns_field="description_vector",
                param=search_params,
                limit=limit,
                output_fields=["id", "document_id", "description"]
            )

            similar_docs = []
            for hits in results:
                for hit in hits:
                    if hit.score >= min_score:
                        similar_docs.append({
                            "id": hit.entity.get("id"),
                            "document_id": hit.entity.get("document_id"),
                            "description": hit.entity.get("description"),
                            "score": hit.score
                        })

            return similar_docs

        except Exception as e:
            print(f"Search error: {e}")
            return []

    async def search_similar_faq(self, query_vector: List[float], limit: int = 10, min_score: float = 0.0) -> List[
        Dict]:
        """Search for similar FAQ questions"""
        try:
            if not self.faq_collection:
                raise Exception("FAQ Collection not initialized")

            if len(query_vector) != self.embedding_dim:
                raise Exception(f"Query vector dimension mismatch: {len(query_vector)} != {self.embedding_dim}")

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            results = self.faq_collection.search(
                data=[query_vector],
                anns_field="question_vector",
                param=search_params,
                limit=limit,
                output_fields=["faq_id", "question", "answer"]
            )

            similar_faqs = []
            for hits in results:
                for hit in hits:
                    if hit.score >= min_score:
                        similar_faqs.append({
                            "faq_id": hit.entity.get("faq_id"),
                            "question": hit.entity.get("question"),
                            "answer": hit.entity.get("answer"),
                            "score": hit.score
                        })

            return similar_faqs

        except Exception as e:
            print(f"FAQ Search error: {e}")
            return []

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            stats = {}

            if self.collection:
                self.collection.load()
                stats["document_count"] = self.collection.num_entities
                stats["document_collection_name"] = self.collection_name
                stats["document_vector_dim"] = self.embedding_dim

            if self.faq_collection:
                self.faq_collection.load()
                stats["faq_count"] = self.faq_collection.num_entities
                stats["faq_collection_name"] = self.faq_collection_name
                stats["faq_vector_dim"] = self.embedding_dim

            return stats

        except Exception as e:
            print(f"Stats error: {e}")
            return {}

    async def health_check(self) -> bool:
        """Check Milvus connection health"""
        try:
            connections.get_connection_addr("default")
            return True
        except:
            return False

    def get_field_limits(self) -> Dict[str, int]:
        """Get field length limits"""
        return {
            "id": self.max_id_length,
            "document_id": self.max_document_id_length,
            "description": self.max_description_length,
            "question": self.max_question_length,
            "answer": self.max_answer_length,
            "embedding_dim": self.embedding_dim
        }


async def main():
    # Khởi tạo MilvusManager
    milvus = MilvusManager(
        host="localhost",
        port="19530",
    )

    # Kết nối & tạo collection
    await milvus.initialize()



if __name__ == "__main__":
    asyncio.run(main())
