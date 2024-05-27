package com.didalgo.ai.gemini.metadata;

import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.Usage;

import java.util.concurrent.ConcurrentHashMap;

public class GeminiChatResponseMetadata extends ConcurrentHashMap<String, Object> implements ChatResponseMetadata {

	private final GeminiUsage usage;

	public GeminiChatResponseMetadata(GeminiUsage usage) {
		this.usage = usage;
	}

	@Override
	public Usage getUsage() {
		return this.usage;
	}

}
