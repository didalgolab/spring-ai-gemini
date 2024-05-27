package com.didalgo.ai.gemini.metadata;

import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.Usage;

public class GeminiChatResponseMetadata implements ChatResponseMetadata {

	private final GeminiUsage usage;

	public GeminiChatResponseMetadata(GeminiUsage usage) {
		this.usage = usage;
	}

	@Override
	public Usage getUsage() {
		return this.usage;
	}

}
