use async_openai::config::OpenAIConfig;
use async_openai::types::CreateEmbeddingRequestArgs;
use async_openai::Client as OpenAIClient;
use regex::Regex;
use std::fs;

use rag::constants::EMBEDDING_MODEL;
use rag::env_load::Config;
use rag::types::{EmbeddingRecord, EmbeddingRecordMetadata};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;

    let text = fs::read_to_string("data/input/faq.txt")?;
    let re = Regex::new(r"\n{2,}|\r\n{2,}")?;
    let chunks: Vec<&str> = re.split(&text).collect();

    let openai_config = OpenAIConfig::new().with_api_key(&config.openai_api_key);
    let openai_client = OpenAIClient::with_config(openai_config);

    let request = CreateEmbeddingRequestArgs::default()
        .model(EMBEDDING_MODEL)
        .input(chunks.clone())
        .build()?;

    let response = openai_client.embeddings().create(request).await?;

    let embeddings: Vec<EmbeddingRecord> = response
        .data
        .iter()
        .enumerate()
        .map(|(i, data)| EmbeddingRecord {
            id: i.to_string(),
            values: data.embedding.clone(),
            metadata: EmbeddingRecordMetadata {
                text: chunks[i].to_string(),
            },
        })
        .collect();

    let json = serde_json::to_string_pretty(&embeddings)?;
    fs::write("data/embeddings/embedding.json", json)?;

    println!("Generated {} embeddings", embeddings.len());

    Ok(())
}
