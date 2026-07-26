#!/bin/bash


# Available models:
# claude-haiku-4.5
# • claude-opus-4.5
# • claude-opus-4.6
# • claude-opus-4.7
# • claude-sonnet-4.5
# • claude-sonnet-4.6
# • deepseek-r1
# • gemini-2.5-flash
# • gemini-2.5-flash-image
# • gemini-2.5-flash-lite
# • gemini-2.5-pro
# • gemini-3-pro-image-preview
# • gemini-3-pro-preview
# • gemini-3.1-flash-image-preview
# • gemini-3.1-flash-lite-preview
# • gemini-3.1-pro-preview
# • gemma-4-31B-it
# • gpt-4.1
# • gpt-4.1-mini
# • gpt-4.1-nano
# • gpt-4o
# • gpt-4o-mini
# • gpt-5
# • gpt-5-chat
# • gpt-5-mini
# • gpt-5-nano
# • gpt-5-pro
# • gpt-5.1
# • gpt-5.2
# • gpt-5.2-chat
# • gpt-5.2-codex
# • gpt-5.3-chat
# • gpt-5.3-codex
# • gpt-5.4
# • gpt-5.4-mini
# • gpt-5.4-nano
# • gpt-5.4-pro
# • gpt-5.5
# • gpt-image-1
# • gpt-image-1-mini
# • gpt-image-1.5
# • gpt-image-2
# • gpt-oss-120b
# • gpt-oss-20b
# • llama-3-1-8b
# • llama-3-2-90b-vision
# • llama-3-3-70b
# • o4-mini
# • qwen-3.5-397b-a17b-FP8
# • text-embedding-3-large
# • text-embedding-3-small

MODELS=(
#    "claude-opus-4.6"
#    "claude-sonnet-4.6"
#    "deepseek-r1"
    "gemini-2.5-flash"
    "gemini-2.5-pro"
    "gemini-3.1-pro-preview"
    "gemma-4-31b-it"
    "gpt-4o"
    "gpt-4o-mini"
    "gpt-5"
    "gpt-5.4-mini"
    "gpt-5.4-nano"
    "gpt-5.4-pro"
    "gpt-oss-120b"
    "gpt-oss-20b"
    "llama-3-1-8b"
    "llama-3-2-90b-vision"
    "llama-3-3-70b"
    "qwen-3.5-397b-a17b-FP8"
)

for model in ${MODELS[@]}; do
    python judge_models.py $model
done
