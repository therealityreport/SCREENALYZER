#!/usr/bin/env python3
"""
Test script for upload functionality.

Demonstrates chunked upload with resume capability.
"""

import io
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.uploads import upload_manager
from screentime.io_utils import validate_cast_image


def create_dummy_video(size_mb: int = 10) -> bytes:
    """Create a dummy file for testing upload."""
    # Create dummy data
    chunk_size = 1024 * 1024  # 1 MB
    data = b"0" * chunk_size
    return data * size_mb


def test_chunked_upload():
    """Test chunked upload with interruption and resume."""
    print("=== Testing Chunked Upload ===\n")

    # Create dummy video data
    print("1. Creating dummy video data (10 MB)...")
    video_data = create_dummy_video(size_mb=10)
    print(f"   Created {len(video_data) / (1024**2):.2f} MB of data\n")

    # Create upload session
    print("2. Creating upload session...")
    session = upload_manager.create_upload_session(
        filename="test_episode.mp4", total_size_bytes=len(video_data)
    )
    print(f"   Session ID: {session.session_id}")
    print(f"   Total chunks: {session.total_chunks}")
    print(f"   Chunk size: {session.chunk_size_bytes / (1024**2):.2f} MB\n")

    # Upload first 3 chunks
    print("3. Uploading first 3 chunks...")
    for chunk_id in range(3):
        start = chunk_id * session.chunk_size_bytes
        end = min(start + session.chunk_size_bytes, len(video_data))
        chunk_data = video_data[start:end]

        result = upload_manager.upload_chunk(session.session_id, chunk_id, chunk_data)
        print(f"   Chunk {chunk_id}: {result['status']} ({result['progress_pct']:.1f}%)")

    print()

    # Simulate interruption - get resume info
    print("4. Simulating interruption - checking resume info...")
    resume_info = upload_manager.resume_upload(session.session_id)
    print(f"   Uploaded chunks: {resume_info['uploaded_chunks']}")
    print(f"   Next chunk: {resume_info['next_chunk_id']}")
    print(f"   Progress: {resume_info['progress_pct']:.1f}%\n")

    # Resume upload - upload remaining chunks
    print("5. Resuming upload...")
    next_chunk = resume_info["next_chunk_id"]
    while next_chunk < session.total_chunks:
        start = next_chunk * session.chunk_size_bytes
        end = min(start + session.chunk_size_bytes, len(video_data))
        chunk_data = video_data[start:end]

        result = upload_manager.upload_chunk(session.session_id, next_chunk, chunk_data)
        print(f"   Chunk {next_chunk}: {result['status']} ({result['progress_pct']:.1f}%)")

        if result["status"] == "completed":
            print(f"\n✅ Upload completed!")
            print(f"   File path: {result['file_path']}")
            break

        next_chunk = result.get("next_chunk_id", next_chunk + 1)

    # Test idempotency - re-upload a chunk
    print("\n6. Testing idempotency - re-uploading chunk 0...")
    start = 0
    end = session.chunk_size_bytes
    chunk_data = video_data[start:end]
    result = upload_manager.upload_chunk(session.session_id, 0, chunk_data)
    print(f"   Result: {result['status']}")

    print("\n=== Upload Test Complete ===\n")


def test_cast_image_validation():
    """Test cast image validation."""
    print("=== Testing Cast Image Validation ===\n")

    # Create a dummy image using PIL
    from PIL import Image

    # Create a 300x300 RGB image
    img = Image.new("RGB", (300, 300), color="red")

    # Save to temp file
    temp_path = Path("data/facebank/test_cast.jpg")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(temp_path, format="JPEG")

    print(f"1. Created test image: {temp_path}")
    print(f"   Size: {img.width}x{img.height}\n")

    # Validate
    print("2. Validating image...")
    is_valid, errors, cast_image = validate_cast_image(temp_path, "TEST_PERSON")

    if is_valid:
        print("   ✅ Validation passed!")
        if cast_image:
            print(f"   Person: {cast_image.person_name}")
            print(f"   Size: {cast_image.width}x{cast_image.height}")
            print(f"   Format: {cast_image.format.value}")
    else:
        print("   ❌ Validation failed!")
        for error in errors:
            print(f"   • {error}")

    # Clean up
    temp_path.unlink(missing_ok=True)

    print("\n=== Cast Image Test Complete ===\n")


if __name__ == "__main__":
    print("\nScreenalyzer Upload Functionality Test\n")
    print("=" * 50)
    print()

    try:
        test_chunked_upload()
        test_cast_image_validation()

        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
