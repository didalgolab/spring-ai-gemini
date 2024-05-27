package com.didalgo.ai.gemini.api;

import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;

import java.util.function.Consumer;

final class ApiUtils {

    public static Consumer<HttpHeaders> getJsonContentHeaders(String apiKey) {
        return (headers) -> {
            headers.set("x-goog-api-key", apiKey);
            headers.setContentType(MediaType.APPLICATION_JSON);
        };
    };
}
