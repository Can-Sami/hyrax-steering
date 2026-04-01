from app.workers.pipeline import InferencePipeline, QwenEmbeddingProvider, WhisperLargeV3Provider


def test_inference_pipeline_returns_transcript_and_embedding() -> None:
    pipeline = InferencePipeline(
        stt_provider=WhisperLargeV3Provider(),
        embedding_provider=QwenEmbeddingProvider(),
    )
    transcript, embedding = pipeline.run(audio_uri='file://sample.wav', language_code='tr')

    assert transcript
    assert len(embedding) == 1024
