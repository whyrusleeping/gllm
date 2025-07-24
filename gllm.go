package gllm

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/whyrusleeping/gollama"
)

type StructuredRequest[T any] struct {
	Model        string
	OutputSpec   *T
	System       string
	Prompt       string
	Context      string
	Images       []string
	MaxToolCalls int
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

type Client struct {
	ollmc *gollama.Client

	Tools map[string]*Tool

	PromptOverride map[string]string
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

func (c *Client) HandleToolCall(call gollama.ToolCall) (string, error) {
	if c.Tools == nil {
		c.Tools = make(map[string]*Tool)
	}
	t, ok := c.Tools[call.Function.Name]
	if !ok {
		return "", fmt.Errorf("no such tool %q", call.Function.Name)
	}

	fmt.Println("CALL: ", t.Name)
	fmt.Println("Arguments: ", call.Function.Arguments)

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

const defaultStructuredCallPrompt = `<context_for_task>
%s
</context_for_task>
When responding, ensure your output matches the following template strictly, output only json, starting with the { character
<output_template>
%s
</output_template>
%s`

func (c *Client) getStructuredCallPrompt() string {
	if c.PromptOverride == nil {
		return defaultStructuredCallPrompt
	}

	v, ok := c.PromptOverride["structured_call"]
	if !ok {
		return defaultStructuredCallPrompt
	}

	return v
}

func ModelCallStructured[T any](c *Client, req *StructuredRequest[T]) (*T, error) {
	ospec, err := renderOutputSpec(req.OutputSpec)
	if err != nil {
		return nil, err
	}

	m := gollama.Message{
		Role:    "user",
		Content: fmt.Sprintf(c.getStructuredCallPrompt(), req.Context, ospec, req.Prompt),
	}

	for _, img := range req.Images {
		m.Images = append(m.Images, img)
	}

	msgs := []gollama.Message{m}

	for {
		glreq := gollama.RequestOptions{
			Model:    req.Model,
			System:   req.System,
			Think:    true,
			Messages: msgs,
		}

		if req.MaxToolCalls > 0 {
			glreq.Tools = c.GetTools()
			glreq.ToolChoice = "auto"
		}

		resp, err := c.ollmc.ChatCompletion(glreq)
		if err != nil {
			return nil, err
		}

		mm := resp.Choices[0].Message

		if len(mm.ToolCalls) == 0 {
			output := cleanJsonOutput(resp.Choices[0].Message.Content)

			var outv T
			if err := json.Unmarshal([]byte(output), &outv); err != nil {
				return nil, err
			}

			return &outv, nil
		}

		if len(mm.ToolCalls) > 1 {
			fmt.Println("MODEL REQUESTED MULTIPLE TOOL CALLS, ONLY DOING ONE")
		}

		toolresp, err := c.HandleToolCall(mm.ToolCalls[0])
		if err != nil {
			return nil, fmt.Errorf("tool call failed: %w", err)
		}

		req.MaxToolCalls--

		msgs = append(msgs,
			mm,
			gollama.Message{
				Role:    "tool",
				Content: toolresp,
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
