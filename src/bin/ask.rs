use async_openai::config::OpenAIConfig;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use async_openai::Client as OpenAIClient;
use std::io::{self, Read};

use rag::constants::{CHAT_MODEL, MAX_CONTEXT_CHARS};
use rag::env_load::Config;
use rag::query_db::query_db;
use rag::types::QueryResult;

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
    let context = build_context(&result);
    let messages = build_messages(&context, prompt);

    let openai_config = OpenAIConfig::new().with_api_key(&config.openai_api_key);
    let openai_client = OpenAIClient::with_config(openai_config);

    let request = CreateChatCompletionRequestArgs::default()
        .model(CHAT_MODEL)
        .messages(messages)
        .build()?;

    let response = openai_client.chat().create(request).await?;
    let content = response.choices[0]
        .message
        .content
        .as_ref()
        .map(|s| s.trim())
        .unwrap_or("");

    println!("{}", content);

    Ok(())
}

fn build_context(result: &QueryResult) -> String {
    let mut sections = Vec::new();

    for (i, m) in result.matches.iter().enumerate() {
        let text = m.metadata.text.trim();
        if text.is_empty() {
            continue;
        }
        sections.push(format!(
            "<document>\n<index>{}</index>\n<score>{:.4}</score>\n<text>{}</text>\n</document>",
            i + 1,
            m.score,
            text
        ));
    }

    let context_raw = format!("<context>\n{}\n</context>", sections.join("\n"));

    if context_raw.len() > MAX_CONTEXT_CHARS {
        context_raw[..MAX_CONTEXT_CHARS].to_string()
    } else {
        context_raw
    }
}

fn build_messages(context: &str, prompt: &str) -> Vec<ChatCompletionRequestMessage> {
    let system_content = "You are a helpful assistant. Use only the provided context to answer the question. Respond in markdown format. If the context is insufficient, say you do not know.";

    let user_content = format!(
        "<request>\n{}\n<question>\n{}\n</question>\n</request>",
        context, prompt
    );

    vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_content)
            .build()
            .unwrap()
            .into(),
        ChatCompletionRequestUserMessageArgs::default()
            .content(user_content)
            .build()
            .unwrap()
            .into(),
    ]
}
