package com.didalgo.ai.gemini.metadata;

import com.didalgo.ai.gemini.api.GeminiApi;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.util.Assert;

public class GeminiUsage implements Usage {

	private final GeminiApi.UsageMetadata usageMetadata;

	public GeminiUsage(GeminiApi.UsageMetadata usageMetadata) {
		Assert.notNull(usageMetadata, "UsageMetadata must not be null");
		this.usageMetadata = usageMetadata;
	}

	@Override
	public Long getPromptTokens() {
		return Long.valueOf(usageMetadata.promptTokenCount());
	}

	@Override
	public Long getGenerationTokens() {
		return Long.valueOf(usageMetadata.candidatesTokenCount());
	}

}
