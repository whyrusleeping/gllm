package gllm

import (
	"bytes"
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
	Tools        []*Tool

	PromptOverride map[string]string
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

type Tool struct {
	Name        string
	Description string
	Params      ToolParams

	Func func(map[string]any) (string, error)
}

// TODO: drop this in favor of the one i'm writing in gollama
type ToolParams struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties"`
	Required   []string       `json:"required"`
}

func (tp *ToolParams) ToMap() map[string]any {
	return map[string]any{
		"type":       tp.Type,
		"properties": tp.Properties,
		"required":   tp.Required,
	}
}

func (tp *ToolParams) ToGollama() gollama.ToolFunctionParams {
	return gollama.ToolFunctionParams{
		Type:       tp.Type,
		Properties: tp.Properties,
		Required:   tp.Required,
	}
}

func (t *Tool) GollamaToolDef() gollama.ToolParam {
	return gollama.ToolParam{
		Function: &gollama.ToolFunction{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Params.ToGollama(),
		},
		Type: "function",
	}
}

func NewClient(olc *gollama.Client) *Client {
	return &Client{
		ollmc: olc,
		Tools: make(map[string]*Tool),
	}
}

type Client struct {
	ollmc *gollama.Client

	Tools map[string]*Tool

	Debug bool
}

func (c *Client) AddTool(t *Tool) {
	if c.Tools == nil {
		c.Tools = make(map[string]*Tool)
	}

	c.Tools[t.Name] = t
}

func (c *Client) GetTools() []gollama.ToolParam {
	var out []gollama.ToolParam
	for _, t := range c.Tools {
		out = append(out, t.GollamaToolDef())
	}

	return out
}

func (c *Client) HandleToolCall(tools []*Tool, call gollama.ToolCall) (string, error) {
	var t *Tool
	for _, ot := range tools {
		if ot.Name == call.Function.Name {
			t = ot
			break
		}
	}
	if t == nil {
		return "", fmt.Errorf("no such tool %q", call.Function.Name)
	}

	if c.Debug {
		fmt.Println("CALL: ", t.Name)
		fmt.Println("Arguments: ", call.Function.Arguments)
	}

	var obj map[string]any
	if err := json.Unmarshal([]byte(call.Function.Arguments), &obj); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	resp, err := t.Func(obj)
	if err != nil {
		return "", fmt.Errorf("tool invocation failed: %w", err)
	}

	return resp, nil
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

func pjson(v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(b))
}

type Response[T any] struct {
	Output        *T
	ModelComment  string
	RawResponse   *gollama.ResponseMessageGenerate
	InputMessages []gollama.Message
}

func ModelCallStructured[T any](c *Client, req *StructuredRequest[T]) (*Response[T], error) {
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

	var tools []*Tool
	var tooldefs []gollama.ToolParam
	for _, t := range c.Tools {
		tools = append(tools, t)
		tooldefs = append(tooldefs, t.GollamaToolDef())
	}
	for _, t := range req.Tools {
		tools = append(tools, t)
		tooldefs = append(tooldefs, t.GollamaToolDef())
	}

	for {
		glreq := gollama.RequestOptions{
			Model:    req.Model,
			System:   req.System,
			Think:    true,
			Messages: msgs,
		}

		if req.MaxToolCalls > 0 {
			glreq.Tools = tooldefs
			glreq.ToolChoice = "auto"
		}

		if c.Debug {
			fmt.Println("Making completion request: ")
			pjson(glreq.Messages)
		}

		resp, err := c.ollmc.ChatCompletion(glreq)
		if err != nil {
			return nil, err
		}

		if c.Debug {
			fmt.Println("Response: ")
			pjson(resp)
		}

		mm := resp.Choices[0].Message

		if len(mm.ToolCalls) == 0 {
			output := cleanJsonOutput(resp.Choices[0].Message.Content)

			if c.Debug {
				fmt.Println("MODEL OUTPUT:")
				fmt.Println(output)
			}

			var message, jsonout string
			if strings.HasPrefix(output, "{") {
				jsonout = output
			} else {
				lines := strings.Split(output, "\n")
				if strings.HasPrefix(lines[len(lines)-1], "{") {
					message = strings.Join(lines[:len(lines)-1], "\n")
					jsonout = lines[len(lines)-1]
				}
			}

			if message != "" {
				fmt.Printf("Model sent a message along with its output: %q", message)
			}
			var outv T
			if err := json.Unmarshal([]byte(jsonout), &outv); err != nil {
				return nil, err
			}

			return &Response[T]{
				Output:        &outv,
				ModelComment:  message,
				RawResponse:   resp,
				InputMessages: msgs,
			}, nil
		}

		if len(mm.ToolCalls) > 1 {
			fmt.Println("MODEL REQUESTED MULTIPLE TOOL CALLS, ONLY DOING ONE")
		}

		fmt.Println("model requested tool call: ", mm.ToolCalls[0])

		toolresp, err := c.HandleToolCall(tools, mm.ToolCalls[0])
		if err != nil {
			return nil, fmt.Errorf("tool call failed: %w", err)
		}

		req.MaxToolCalls--

		msgs = append(msgs,
			mm,
			gollama.Message{
				Role:       "tool",
				Content:    toolresp,
				ToolCallID: mm.ToolCalls[0].ID,
			},
		)
	}
}

func cleanJsonOutput(s string) string {
	output := strings.TrimSpace(s)
	output = strings.TrimPrefix(output, "```json")
	output = strings.TrimSuffix(output, "```")
	output = strings.TrimSpace(output)

	return output
}
