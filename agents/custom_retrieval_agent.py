from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import QueryType
import logging

logger = logging.getLogger(__name__)

class CustomRetrievalAgent:
    def __init__(self, search_endpoint, api_key, index_name, vector_field_name="embedding"):
        """
        Initialize the retrieval agent for Azure AI Search.

        Args:
            search_endpoint (str): Azure AI Search endpoint.
            api_key (str): Azure AI Search Admin Key.
            index_name (str): Name of the search index.
            vector_field_name (str): Name of the vector field in the index (default: "embedding").
        """
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
        self.vector_field_name = vector_field_name
        logger.info(f"CustomRetrievalAgent initialized with endpoint: {search_endpoint} and index: {index_name}")

    async def retrieve_by_text(self, query_text: str, top_k: int = 5, fields: list = None):
        """
        Retrieve documents by text query.

        Args:
            query_text (str): The search query text.
            top_k (int): Number of top results to return.
            fields (list): List of fields to include in the response.

        Returns:
            list: List of documents matching the query.
        """
        if not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        fields = fields or ["HotelId", "HotelName", "Description", "Category", "Tags", "Address", "Rooms"]
        try:
            logger.info(f"Performing search with query: '{query_text}' and fields: {fields}")
            results = self.search_client.search(
                search_text=query_text,
                top=top_k,
                select=fields,
                query_type=QueryType.SIMPLE
            )
            documents = [
                {field: result.get(field) for field in fields if field in result}
                for result in results
            ]
            logger.debug(f"Retrieved documents: {documents}")
            return documents
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}", exc_info=True)
            raise RuntimeError("Failed to retrieve documents by text query.") from e

    async def retrieve_by_filter(self, filter_query: str, top_k: int = 5, fields: list = None):
        """
        Retrieve documents by applying a filter query.

        Args:
            filter_query (str): OData filter string for filtering results.
            top_k (int): Number of top results to return.
            fields (list): List of fields to include in the response.

        Returns:
            list: List of documents matching the filter query.
        """
        fields = fields or ["HotelId", "HotelName", "Description", "Tags", "Address"]
        try:
            logger.info(f"Performing filtered search with query: '{filter_query}'")
            results = self.search_client.search(
                search_text="*",  # Empty search text with filter
                filter=filter_query,
                top=top_k,
                select=fields
            )
            documents = [
                {field: result.get(field) for field in fields if field in result}
                for result in results
            ]
            logger.debug(f"Retrieved filtered documents: {documents}")
            return documents
        except Exception as e:
            logger.error(f"Error during filtered retrieval: {e}", exc_info=True)
            raise RuntimeError("Failed to retrieve documents by filter.") from e

    async def retrieve_all_fields(self, document_ids: list[str]):
        """
        Retrieve all fields for a list of document IDs.

        Args:
            document_ids (list[str]): List of document IDs to fetch.

        Returns:
            list: List of documents with all fields.
        """
        try:
            logger.info(f"Fetching all fields for document IDs: {document_ids}")
            documents = []
            for doc_id in document_ids:
                result = self.search_client.get_document(doc_id)
                documents.append(result)
            logger.debug(f"Retrieved full documents: {documents}")
            return documents
        except Exception as e:
            logger.error(f"Error during retrieval of all fields: {e}", exc_info=True)
            raise RuntimeError("Failed to retrieve all fields for documents.") from e
