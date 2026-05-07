package provider

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBedrockResponsePreservesClaudeNativeThinkingSignature(t *testing.T) {
	provider := &bedrockProvider{}
	ctx := newMockMultipartHttpContext()
	ctx.SetContext("needClaudeResponseConversion", true)

	response := provider.buildChatCompletionResponse(ctx, &bedrockConverseResponse{
		Output: converseOutputMemberMessage{Message: message{
			Role: roleAssistant,
			Content: []contentBlock{
				{ReasoningContent: &reasoningContent{ReasoningText: reasoningText{Text: "reasoning", Signature: "sig"}}},
				{Text: "answer"},
			},
		}},
		StopReason: "end_turn",
	})
	body, err := json.Marshal(response)
	require.NoError(t, err)

	converted, err := (&ClaudeToOpenAIConverter{}).ConvertOpenAIResponseToClaude(ctx, body)
	require.NoError(t, err)

	var claudeResponse claudeTextGenResponse
	require.NoError(t, json.Unmarshal(converted, &claudeResponse))
	require.Len(t, claudeResponse.Content, 2)
	assert.Equal(t, "thinking", claudeResponse.Content[0].Type)
	require.NotNil(t, claudeResponse.Content[0].Thinking)
	require.NotNil(t, claudeResponse.Content[0].Signature)
	assert.Equal(t, "reasoning", *claudeResponse.Content[0].Thinking)
	assert.Equal(t, "sig", *claudeResponse.Content[0].Signature)
	assert.Equal(t, "text", claudeResponse.Content[1].Type)
	require.NotNil(t, claudeResponse.Content[1].Text)
	assert.Equal(t, "answer", *claudeResponse.Content[1].Text)
}

func TestBedrockStreamPreservesClaudeNativeThinkingSignature(t *testing.T) {
	provider := &bedrockProvider{}
	ctx := newMockMultipartHttpContext()
	ctx.SetContext("needClaudeResponseConversion", true)
	converter := &ClaudeToOpenAIConverter{}

	textChunk, err := provider.convertEventFromBedrockToOpenAI(ctx, ConverseStreamEvent{
		ContentBlockIndex: 0,
		Delta: &converseStreamEventContentBlockDelta{
			ReasoningContent: &reasoningContentDelta{Text: "reasoning"},
		},
	})
	require.NoError(t, err)
	_, err = converter.ConvertOpenAIStreamResponseToClaude(ctx, textChunk)
	require.NoError(t, err)

	signatureChunk, err := provider.convertEventFromBedrockToOpenAI(ctx, ConverseStreamEvent{
		ContentBlockIndex: 0,
		Delta: &converseStreamEventContentBlockDelta{
			ReasoningContent: &reasoningContentDelta{Signature: "sig"},
		},
	})
	require.NoError(t, err)
	converted, err := converter.ConvertOpenAIStreamResponseToClaude(ctx, signatureChunk)
	require.NoError(t, err)

	events := parseClaudeSSEEvents(t, converted)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_delta", events[0].Name)
	require.NotNil(t, events[0].Payload.Delta)
	assert.Equal(t, "signature_delta", events[0].Payload.Delta.Type)
	assert.Equal(t, "sig", events[0].Payload.Delta.Signature)
}

func TestBedrockStreamPreservesClaudeNativeIndexesAndStops(t *testing.T) {
	provider := &bedrockProvider{}
	ctx := newMockMultipartHttpContext()
	ctx.SetContext("needClaudeResponseConversion", true)
	converter := &ClaudeToOpenAIConverter{}

	chunk, err := provider.convertEventFromBedrockToOpenAI(ctx, ConverseStreamEvent{
		ContentBlockIndex: 2,
		Delta: &converseStreamEventContentBlockDelta{
			ReasoningContent: &reasoningContentDelta{Text: "reasoning"},
		},
	})
	require.NoError(t, err)
	converted, err := converter.ConvertOpenAIStreamResponseToClaude(ctx, chunk)
	require.NoError(t, err)
	events := parseClaudeSSEEvents(t, converted)
	require.Len(t, events, 2)
	assert.Equal(t, "content_block_start", events[0].Name)
	require.NotNil(t, events[0].Payload.Index)
	assert.Equal(t, 2, *events[0].Payload.Index)
	assert.Equal(t, "content_block_delta", events[1].Name)
	require.NotNil(t, events[1].Payload.Index)
	assert.Equal(t, 2, *events[1].Payload.Index)

	chunk, err = provider.convertEventFromBedrockToOpenAI(ctx, ConverseStreamEvent{
		ContentBlockIndex: 2,
		ContentBlockStop:  &contentBlockStop{ContentBlockIndex: 2},
	})
	require.NoError(t, err)
	converted, err = converter.ConvertOpenAIStreamResponseToClaude(ctx, chunk)
	require.NoError(t, err)
	events = parseClaudeSSEEvents(t, converted)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_stop", events[0].Name)
	require.NotNil(t, events[0].Payload.Index)
	assert.Equal(t, 2, *events[0].Payload.Index)
}

func TestBedrockResponsePreservesClaudeNativeRedactedThinking(t *testing.T) {
	provider := &bedrockProvider{}
	ctx := newMockMultipartHttpContext()
	ctx.SetContext("needClaudeResponseConversion", true)

	response := provider.buildChatCompletionResponse(ctx, &bedrockConverseResponse{
		Output: converseOutputMemberMessage{Message: message{
			Role: roleAssistant,
			Content: []contentBlock{
				{ReasoningContent: &reasoningContent{RedactedContent: "opaque-base64"}},
				{Text: "answer"},
			},
		}},
		StopReason: "end_turn",
	})
	body, err := json.Marshal(response)
	require.NoError(t, err)

	converted, err := (&ClaudeToOpenAIConverter{}).ConvertOpenAIResponseToClaude(ctx, body)
	require.NoError(t, err)

	var claudeResponse claudeTextGenResponse
	require.NoError(t, json.Unmarshal(converted, &claudeResponse))
	require.Len(t, claudeResponse.Content, 2)
	assert.Equal(t, "redacted_thinking", claudeResponse.Content[0].Type)
	assert.Equal(t, "opaque-base64", claudeResponse.Content[0].Data)
}

func TestBedrockRequestPreservesClaudeNativeThinkingAndToolResult(t *testing.T) {
	provider := &bedrockProvider{}
	ctx := newMockMultipartHttpContext()
	ctx.SetContext(ctxKeyClaudeNativeRequestBody, []byte(`{
		"model":"claude",
		"messages":[{
			"role":"assistant",
			"content":[
				{"type":"thinking","thinking":"reasoning","signature":"sig"},
				{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"query":"q"}}
			]
		},{
			"role":"user",
			"content":[{
				"type":"tool_result",
				"tool_use_id":"toolu_1",
				"is_error":true,
				"content":[{"type":"text","text":"failed"}]
			}]
		}]
	}`))

	body, err := provider.buildBedrockTextGenerationRequest(&chatCompletionRequest{
		Model:     "claude",
		MaxTokens: 1024,
		Messages:  []chatMessage{{Role: roleUser, Content: "placeholder"}},
	}, provider.getClaudeNativeRequest(ctx), nil)
	require.NoError(t, err)

	var request bedrockTextGenRequest
	require.NoError(t, json.Unmarshal(body, &request))
	require.Len(t, request.Messages, 2)
	require.Len(t, request.Messages[0].Content, 2)
	require.NotNil(t, request.Messages[0].Content[0].ReasoningContent)
	assert.Equal(t, "reasoning", request.Messages[0].Content[0].ReasoningContent.ReasoningText.Text)
	assert.Equal(t, "sig", request.Messages[0].Content[0].ReasoningContent.ReasoningText.Signature)
	require.NotNil(t, request.Messages[0].Content[1].ToolUse)
	assert.Equal(t, "toolu_1", request.Messages[0].Content[1].ToolUse.ToolUseId)
	require.NotNil(t, request.Messages[1].Content[0].ToolResult)
	assert.Equal(t, "error", request.Messages[1].Content[0].ToolResult.Status)
	assert.Equal(t, "failed", request.Messages[1].Content[0].ToolResult.Content[0].Text)
}

func TestBedrockResponseUsesReasoningContentInsteadOfThinkTags(t *testing.T) {
	provider := &bedrockProvider{}

	response := provider.buildChatCompletionResponse(newMockMultipartHttpContext(), &bedrockConverseResponse{
		Output: converseOutputMemberMessage{Message: message{
			Role: roleAssistant,
			Content: []contentBlock{
				{ReasoningContent: &reasoningContent{ReasoningText: reasoningText{Text: "reasoning"}}},
				{Text: "answer"},
			},
		}},
		StopReason: "end_turn",
	})

	require.Len(t, response.Choices, 1)
	require.NotNil(t, response.Choices[0].Message)
	assert.Equal(t, "reasoning", response.Choices[0].Message.ReasoningContent)
	assert.Equal(t, "answer", response.Choices[0].Message.Content)
	assert.NotContains(t, response.Choices[0].Message.StringContent(), "<think>")
}

func TestBedrockStreamUsesReasoningContentInsteadOfThinkTags(t *testing.T) {
	provider := &bedrockProvider{}
	ctx := newMockMultipartHttpContext()

	chunk, err := provider.convertEventFromBedrockToOpenAI(ctx, ConverseStreamEvent{
		ContentBlockIndex: 0,
		Delta: &converseStreamEventContentBlockDelta{
			ReasoningContent: &reasoningContentDelta{Text: "reasoning"},
		},
	})
	require.NoError(t, err)

	body := strings.TrimPrefix(strings.TrimSpace(string(chunk)), ssePrefix)
	var response chatCompletionResponse
	require.NoError(t, json.Unmarshal([]byte(body), &response))
	require.Len(t, response.Choices, 1)
	require.NotNil(t, response.Choices[0].Delta)
	assert.Equal(t, "reasoning", response.Choices[0].Delta.ReasoningContent)
	assert.Nil(t, response.Choices[0].Delta.Content)
}
