use std::io::{self, Read};

use rag::env_load::Config;
use rag::query_db::query_db;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;

    let mut prompt_raw = String::new();
    io::stdin().read_to_string(&mut prompt_raw)?;
    let prompt = prompt_raw.trim();

    if prompt.is_empty() {
        return Err("Prompt is empty".into());
    }

    let result = query_db(prompt, &config).await?;
    let json = serde_json::to_string_pretty(&result)?;
    println!("{}", json);

    Ok(())
}
