package gllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"text/template"

	"github.com/whyrusleeping/gollama"
)

type StructuredRequest[T any] struct {
	Model string

	// System prompt for the LLM
	System string

	// Prompt for this request, essentially the function definition
	Prompt string

	// Context is the input for this specific task, if you are making
	// multiple calls with different inputs, this is the field that should
	// vary
	Context string

	// MessagePrefill sets the chat history before this prompt gets
	// executed. This will override the system prompt for this request, so
	// if you need the system prompt ensure that its included here
	MessagePrefill []gollama.Message

	// Images is an array of base64 encoded images, no other prefixing
	Images []string

	MaxToolCalls int
	Tools        []*gollama.Tool

	PromptOverride map[string]string

	// Think enables extended thinking/reasoning mode if supported by the model.
	// When nil, defaults to true for backwards compatibility.
	// Use BoolPtr(false) to disable thinking.
	Think *bool
}

// BoolPtr returns a pointer to a bool value, useful for setting Think field
func BoolPtr(b bool) *bool {
	return &b
}

func renderOutputSpec(obj any) (string, error) {
	if cos, ok := obj.(customOutputSpec); ok {
		return cos.DescribeType(), nil
	}

	b, err := json.Marshal(obj)
	return string(b), err
}

type customOutputSpec interface {
	DescribeType() string
}

func NewClient(olc *gollama.Client) *Client {
	return &Client{
		ollmc: olc,
	}
}

type Client struct {
	ollmc *gollama.Client

	// Debug enables debug output. If DebugFunc is nil, uses fmt.Println.
	Debug bool

	// DebugFunc is called for debug output when Debug is true.
	// If nil, defaults to fmt.Println.
	DebugFunc func(format string, args ...any)
}

// debugf outputs debug information if Debug is enabled
func (c *Client) debugf(format string, args ...any) {
	if !c.Debug {
		return
	}
	if c.DebugFunc != nil {
		c.DebugFunc(format, args...)
	} else {
		fmt.Printf(format+"\n", args...)
	}
}

// debugJSON outputs JSON-formatted debug information
func (c *Client) debugJSON(v any) {
	if !c.Debug {
		return
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		c.debugf("(failed to marshal: %v)", err)
		return
	}
	c.debugf("%s", string(b))
}

type structuredCallParams struct {
	OutputTemplate string
	Prompt         string
	Context        string
	MaxToolCalls   int
}

const defaultStructuredCallPrompt = `
When responding, ensure your output matches the following template strictly, output only json, starting with the { character
<output_template>
{{.OutputTemplate}}
</output_template>
{{.Prompt}}
{{ if gt .MaxToolCalls 0 }}
Max at most {{.MaxToolCalls}} tool calls.
{{end}}
<context_for_task>
{{.Context}}
</context_for_task>`

const (
	PromptTypeStructuredCall = "structured_call"
)

func (r *StructuredRequest[T]) getStructuredCallPrompt() string {
	if r.PromptOverride == nil {
		return defaultStructuredCallPrompt
	}

	v, ok := r.PromptOverride[PromptTypeStructuredCall]
	if !ok {
		return defaultStructuredCallPrompt
	}

	return v
}

// EstimateRequestSize estimates the size in bytes of this request when converted to a batch request.
// This is useful for checking against the batch API limits (256 MB per batch).
// The estimate includes the JSON serialization of the entire request structure.
func (r *StructuredRequest[T]) EstimateRequestSize() (int, error) {
	// Get the output spec to include in the prompt
	ospec, err := renderOutputSpec(new(T))
	if err != nil {
		return 0, fmt.Errorf("failed to render output spec: %w", err)
	}

	// Build the prompt using the template
	templ, err := template.New("prompt").Parse(r.getStructuredCallPrompt())
	if err != nil {
		return 0, fmt.Errorf("failed to parse prompt template: %w", err)
	}

	buf := new(bytes.Buffer)
	if err := templ.Execute(buf, &structuredCallParams{
		OutputTemplate: ospec,
		Prompt:         r.Prompt,
		Context:        r.Context,
		MaxToolCalls:   r.MaxToolCalls,
	}); err != nil {
		return 0, fmt.Errorf("prompt template execution failed: %w", err)
	}

	// Build the messages array as it would be in the batch request
	var msgs []gollama.Message
	if len(r.MessagePrefill) > 0 {
		msgs = r.MessagePrefill
	} else if r.System != "" {
		msgs = append(msgs, gollama.Message{
			Role:    "system",
			Content: r.System,
		})
	}

	userMsg := gollama.Message{
		Role:    "user",
		Content: buf.String(),
		Images:  r.Images,
	}
	msgs = append(msgs, userMsg)

	// Create a batch request params structure to estimate size
	batchReq := gollama.BatchRequest{
		CustomID: "estimate",
		Params: gollama.BatchRequestParams{
			Model:     r.Model,
			MaxTokens: 4096,
			Messages:  msgs,
		},
	}

	// Serialize to JSON to get the actual size
	data, err := json.Marshal(batchReq)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal request: %w", err)
	}

	return len(data), nil
}

type Response[T any] struct {
	Output        *T
	ModelComment  string
	RawResponse   *gollama.ResponseMessageGenerate
	InputMessages []gollama.Message
}

// BatchResponse represents the result of a batch of structured requests
type BatchResponse[T any] struct {
	BatchID       string
	Status        string
	RequestCounts gollama.BatchRequestCounts
	Results       []*BatchResult[T] // Available when batch is completed
	RawBatch      *gollama.Batch
}

// BatchResult represents a single result from a batch
type BatchResult[T any] struct {
	CustomID     string
	Output       *T
	ModelComment string
	Error        *gollama.BatchError
	ResultType   string // "succeeded", "errored", "canceled", "expired"
}

// ModelCallStructured makes a structured LLM call that returns a typed response.
// The context is used for tool calls and can be used for cancellation.
func ModelCallStructured[T any](c *Client, ctx context.Context, req *StructuredRequest[T]) (*Response[T], error) {
	if ctx == nil {
		ctx = context.Background()
	}
	ospec, err := renderOutputSpec(new(T))
	if err != nil {
		return nil, err
	}

	// system
	var msgs []gollama.Message
	if len(req.MessagePrefill) > 0 {
		msgs = req.MessagePrefill
	} else if req.System != "" {
		msgs = append(msgs, gollama.Message{
			Role:    "system",
			Content: req.System,
		})
	}

	templ, err := template.New("prompt").Parse(req.getStructuredCallPrompt())
	if err != nil {
		return nil, fmt.Errorf("failed to parse prompt template: %w", err)
	}

	buf := new(bytes.Buffer)
	if err := templ.Execute(buf, &structuredCallParams{
		OutputTemplate: ospec,
		Prompt:         req.Prompt,
		Context:        req.Context,
		MaxToolCalls:   req.MaxToolCalls,
	}); err != nil {
		return nil, fmt.Errorf("prompt template execution failed: %w", err)
	}

	m := gollama.Message{
		Role:    "user",
		Content: buf.String(),
	}

	for _, img := range req.Images {
		m.Images = append(m.Images, img)
	}

	msgs = append(msgs, m)

	// Build tool definitions for API request
	var tooldefs []gollama.ToolParam
	for _, t := range req.Tools {
		tooldefs = append(tooldefs, t.ApiDef())
	}

	// Default Think to true for backwards compatibility
	think := true
	if req.Think != nil {
		think = *req.Think
	}

	for {
		glreq := gollama.RequestOptions{
			Model:    req.Model,
			System:   req.System,
			Think:    think,
			Messages: msgs,
		}

		if req.MaxToolCalls > 0 {
			glreq.Tools = tooldefs
			glreq.ToolChoice = "auto"
		}

		if c.Debug {
			c.debugf("Making completion request:")
			c.debugJSON(glreq.Messages)
		}

		resp, err := c.ollmc.ChatCompletion(glreq)
		if err != nil {
			return nil, err
		}

		if c.Debug {
			c.debugf("Response:")
			c.debugJSON(resp)
		}

		mm := resp.Choices[0].Message

		if len(mm.ToolCalls) == 0 {
			output := cleanJsonOutput(resp.Choices[0].Message.Content)

			c.debugf("MODEL OUTPUT:\n%s", output)

			message, jsonout := extractJSONAndComment(output)

			if message != "" {
				c.debugf("Model sent a message along with its output: %q", message)
			}
			var outv T
			if err := json.Unmarshal([]byte(jsonout), &outv); err != nil {
				return nil, fmt.Errorf("failed to parse JSON output: %w (output was: %s)", err, output)
			}

			return &Response[T]{
				Output:        &outv,
				ModelComment:  message,
				RawResponse:   resp,
				InputMessages: msgs,
			}, nil
		}

		c.debugf("Model requested %d tool call(s)", len(mm.ToolCalls))

		// Add the assistant message with tool calls
		msgs = append(msgs, mm)

		// Handle all tool calls
		for _, tc := range mm.ToolCalls {
			c.debugf("Tool call: %s %s", tc.Function.Name, tc.Function.Arguments)

			toolresp, err := gollama.HandleToolCall(ctx, req.Tools, tc)
			if err != nil {
				// Return error to model instead of failing completely
				toolresp = fmt.Sprintf("Error: %v", err)
				c.debugf("Tool call error (sending to model): %s", toolresp)
			}

			msgs = append(msgs, gollama.Message{
				Role:       "tool",
				Content:    toolresp,
				ToolCallID: tc.ID,
			})

			req.MaxToolCalls--
			if req.MaxToolCalls <= 0 {
				break
			}
		}
	}
}

func cleanJsonOutput(s string) string {
	output := strings.TrimSpace(s)
	output = strings.TrimPrefix(output, "```json")
	output = strings.TrimSuffix(output, "```")
	output = strings.TrimSpace(output)

	return output
}

// extractJSONAndComment separates any text before the JSON from the JSON itself
// Returns (comment, json) where comment is any text before the opening brace
func extractJSONAndComment(output string) (string, string) {
	if strings.HasPrefix(output, "{") {
		return "", output
	}

	lines := strings.Split(output, "\n")
	for i, l := range lines {
		if strings.HasPrefix(l, "{") {
			comment := strings.TrimSpace(strings.Join(lines[:i], "\n"))
			jsonout := strings.Join(lines[i:], "\n")
			return comment, jsonout
		}
	}

	// No JSON found, return everything as comment
	return output, ""
}

// ModelCallStructuredBatch creates a batch of structured requests and submits them to the API
// Returns a BatchResponse containing the batch ID and status. Use GetModelCallBatchResults to retrieve results.
// Note: Tool calling is not supported in batch mode.
func ModelCallStructuredBatch[T any](c *Client, model string, requests []*StructuredRequest[T]) (*BatchResponse[T], error) {
	if len(requests) == 0 {
		return nil, fmt.Errorf("no requests provided")
	}

	// Get output spec from the generic type
	ospec, err := renderOutputSpec(new(T))
	if err != nil {
		return nil, fmt.Errorf("failed to render output spec: %w", err)
	}

	// Convert each StructuredRequest into a BatchRequest
	var batchRequests []gollama.BatchRequest
	for i, req := range requests {
		// Build the prompt using the template
		templ, err := template.New("prompt").Parse(req.getStructuredCallPrompt())
		if err != nil {
			return nil, fmt.Errorf("failed to parse prompt template for request %d: %w", i, err)
		}

		buf := new(bytes.Buffer)
		if err := templ.Execute(buf, &structuredCallParams{
			OutputTemplate: ospec,
			Prompt:         req.Prompt,
			Context:        req.Context,
			MaxToolCalls:   0, // Tool calling not supported in batch mode
		}); err != nil {
			return nil, fmt.Errorf("prompt template execution failed for request %d: %w", i, err)
		}

		// Build messages array
		var msgs []gollama.Message
		if len(req.MessagePrefill) > 0 {
			msgs = req.MessagePrefill
		} else if req.System != "" {
			msgs = append(msgs, gollama.Message{
				Role:    "system",
				Content: req.System,
			})
		}

		userMsg := gollama.Message{
			Role:    "user",
			Content: buf.String(),
		}

		// Add images if present
		for _, img := range req.Images {
			userMsg.Images = append(userMsg.Images, img)
		}

		msgs = append(msgs, userMsg)

		// Create batch request with custom ID
		customID := fmt.Sprintf("request-%d", i)
		batchRequests = append(batchRequests, gollama.BatchRequest{
			CustomID: customID,
			Params: gollama.BatchRequestParams{
				Model:     model,
				MaxTokens: 4096, // Default, can be made configurable
				Messages:  msgs,
			},
		})
	}

	// Submit the batch
	batch, err := c.ollmc.CreateBatch(gollama.CreateBatchRequest{
		Requests: batchRequests,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create batch: %w", err)
	}

	return &BatchResponse[T]{
		BatchID:       batch.ID,
		Status:        batch.ProcessingStatus,
		RequestCounts: batch.RequestCounts,
		RawBatch:      batch,
	}, nil
}

// GetModelCallBatchResults retrieves and parses the results of a completed batch
func GetModelCallBatchResults[T any](c *Client, batchID string) (*BatchResponse[T], error) {
	// First get the batch status
	batch, err := c.ollmc.GetBatch(batchID)
	if err != nil {
		return nil, fmt.Errorf("failed to get batch status: %w", err)
	}

	response := &BatchResponse[T]{
		BatchID:       batch.ID,
		Status:        batch.ProcessingStatus,
		RequestCounts: batch.RequestCounts,
		RawBatch:      batch,
	}

	// If batch is not ended, return without results
	if batch.ProcessingStatus != "ended" {
		return response, nil
	}

	// Get the results
	results, err := c.ollmc.GetBatchResults(batchID)
	if err != nil {
		return nil, fmt.Errorf("failed to get batch results: %w", err)
	}

	// Parse each result
	var batchResults []*BatchResult[T]
	for _, result := range results {
		br := &BatchResult[T]{
			CustomID:   result.CustomID,
			ResultType: result.Result.Type,
		}

		switch result.Result.Type {
		case "succeeded":
			if result.Result.Message == nil {
				br.Error = &gollama.BatchError{
					Type:    "missing_message",
					Message: "result marked as succeeded but message is nil",
				}
			} else if len(result.Result.Message.Content) == 0 {
				br.Error = &gollama.BatchError{
					Type:    "missing_content",
					Message: "result marked as succeeded but no content returned",
				}
			} else {
				// Extract text from content blocks
				var contentText string
				for _, block := range result.Result.Message.Content {
					if block.Type == "text" {
						contentText += block.Text
					}
				}

				output := cleanJsonOutput(contentText)
				message, jsonout := extractJSONAndComment(output)
				br.ModelComment = message

				if jsonout == "" {
					br.Error = &gollama.BatchError{
						Type:    "no_json_output",
						Message: "no JSON output found in response",
					}
				} else {
					var outv T
					if err := json.Unmarshal([]byte(jsonout), &outv); err != nil {
						// If parsing fails, store the error but continue processing other results
						br.Error = &gollama.BatchError{
							Type:    "parsing_error",
							Message: fmt.Sprintf("failed to parse JSON output: %v", err),
						}
					} else {
						br.Output = &outv
					}
				}
			}

		case "errored":
			br.Error = result.Result.Error

		case "canceled", "expired":
			// No additional processing needed, type is already set
		}

		batchResults = append(batchResults, br)
	}

	response.Results = batchResults
	return response, nil
}
