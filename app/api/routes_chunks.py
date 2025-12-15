"""
API routes for chunk insertion and retrieval.

This module provides endpoints for:
- Inserting text chunks into the vector database
- Retrieving similar chunks based on query text
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List

from app.models.schemas import ChunkInsert, ChunkResponse, ChunkMetadata, ChunkInsertBatch
from app.services.chunk_store import ChunkStore
from app.api.dependencies import get_chunk_store
from app.core.logging import log_api_call, API_LOGGER
from app.core.context import request_id_var

chunk_router = APIRouter(prefix="/v1/chunks", tags=["chunks"])

@chunk_router.post(
    "/insert",
    response_model=ChunkResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert a chunk into the vector database",
    responses={
        201: {"description": "Chunk successfully inserted"},
        400: {"description": "Chunk exceeds maximum length"},
        500: {"description": "Database insertion failed"}
    }
)

@log_api_call
def insert_chunk(
    chunk_data: ChunkInsert, store: ChunkStore = Depends(get_chunk_store)
) -> ChunkResponse:
    """
    Insert a text chunk into the vector database.

    Args:
        chunk_data: ChunkInsert containing text and metadata

    Returns:
        ChunkResponse with chunk ID and details
    """
    try:
        metadata_dict = None
        if chunk_data.metadata:
            metadata_dict = {
                "source": chunk_data.metadata.source,
                "position": chunk_data.metadata.position
            }
        
        chunk = store.insert_chunk(
            text=chunk_data.text,
            metadata=metadata_dict
        )

        return ChunkResponse(
            id=chunk.id,
            text=chunk.text,
            metadata=ChunkMetadata(
                length=chunk.metadata.get("length", 0),
                source=chunk.metadata.get("source"),
                position=chunk.metadata.get("position")
            ),
            distance=None,
            request_id=request_id_var.get()
        )

    except ValueError as e:
        API_LOGGER.error(f"[insert_chunk] Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e

    except RuntimeError as e:
        API_LOGGER.error(f"[insert_chunk] Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to insert chunk: {str(e)}"
        ) from e

    except Exception as e:
        API_LOGGER.error(f"[insert_chunk] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during chunk insertion"
        ) from e

@chunk_router.post(
    "/insert_batch",
    response_model=List[ChunkResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Insert a batch of chunks into the vector database",
)

@log_api_call
def insert_chunk_batch(
    batch_data: "ChunkInsertBatch", store: ChunkStore = Depends(get_chunk_store)
) -> List[ChunkResponse]:
    """
    Insert a batch of text chunks into the vector database.
    """
    try:
        texts = [chunk.text for chunk in batch_data.chunks]
        metadatas = []
        for chunk in batch_data.chunks:
            if chunk.metadata:
                meta_dict = {k: v for k, v in chunk.metadata.model_dump().items() if v is not None}
                metadatas.append(meta_dict)
            else:
                metadatas.append({})

        inserted_chunks = store.insert_chunks(texts=texts, metadatas=metadatas)

        request_id = request_id_var.get()
        return [
            ChunkResponse(
                id=chunk.id,
                text=chunk.text,
                metadata=ChunkMetadata(
                    length=chunk.metadata.get("length", 0),
                    source=chunk.metadata.get("source"),
                    position=chunk.metadata.get("position")
                ),
                distance=None,
                request_id=request_id
            ) for chunk in inserted_chunks
        ]
    except ValueError as e:
        API_LOGGER.error(f"[insert_chunk_batch] Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except RuntimeError as e:
        API_LOGGER.error(f"[insert_chunk_batch] Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to insert chunk batch: {str(e)}"
        ) from e

@chunk_router.get(
    "/retrieve",
    response_model=List[ChunkResponse],
    summary="Retrieve similar chunks from the vector database",
    responses={
        200: {"description": "Successfully retrieved chunks"},
        400: {"description": "Invalid query parameters"},
        500: {"description": "Database query failed"}
    }
)

@log_api_call
def retrieve_chunks(
    text: str, n_chunks: int = 5, store: ChunkStore = Depends(get_chunk_store)
) -> List[ChunkResponse]:
    """
    Retrieve the most similar chunks to the query text.

    Args:
        text: Query text to find similar chunks
        n_chunks: Number of chunks to retrieve (default: 5, max: 100)

    Returns:
        List of ChunkResponse objects ordered by similarity

    Raises:
        HTTPException 400: If parameters are invalid
        HTTPException 500: If database query fails
    """
    try:
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")

        if n_chunks < 1 or n_chunks > 100: # magic numbers no good
            raise ValueError("n_chunks must be between 1 and 100")

        chunks = store.retrieve_chunks(query_text=text, n_chunks=n_chunks)

        request_id = request_id_var.get()
        return [
            ChunkResponse(
                id=chunk.id,
                text=chunk.text,
                metadata=ChunkMetadata(
                    length=chunk.metadata.get("length", 0),
                    source=chunk.metadata.get("source"),
                    position=chunk.metadata.get("position")
                ),
                distance=chunk.metadata.get('distance'),
                request_id=request_id
            )
            for chunk in chunks
        ]

    except ValueError as e:
        API_LOGGER.error(f"[retrieve_chunks] Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e

    except RuntimeError as e:
        API_LOGGER.error(f"[retrieve_chunks] Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}"
        ) from e

    except Exception as e:
        API_LOGGER.error(f"[retrieve_chunks] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during chunk retrieval"
        ) from e
