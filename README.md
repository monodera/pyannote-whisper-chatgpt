# pyannote-whisper-chatgpt

A Python package to transcribe speech by Whisper with diarization (speaker identification) using pyannote.audio and send the results to OpenAI Chat API to generate, for example, the summary of the conversation.

My primary interest is to automatically make the meeting minutes from recorded audio.

## Requirements

The following packages and their dependencies are required. Some packages need to be specific versions.

- Python (>= 3.9, < 3.11)
- Pydub
- pyannote.audio
- PyTorch (1.11.0)
- torchvision (0.12.0)
- torchaudio (0.11.0)
- pandas
- hmmlearn
- openai-whisper
- pyannote-whisper
- openai
- tiktoken
- logzero
- tenacity (to be used, not currently)

## Installation

I'm currently using PDM for package management, but should be also installed by setuptools, etc.

```
git clone <repository>
cd <repository>
# pyenv local 3.9 # optional
pdm install
```

### Environment variables

In macOS, I had to set the following environment variablesThese two lines need to compile `pyannote-whisper`.

```bash
# to compile pyannote-whisper
export CPATH=/opt/homebrew/include
export LIBRARY_PATH=/opt/homebrew/lib
```


```bash
# to run pyannote-audio
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
```

## Usage

### Hugging Face

In order to use `pyannote.audio`, you need to get an access token for [Hugging Face](https://huggingface.co/). Please follow the following steps.

1. visit hf.co/pyannote/speaker-diarization and accept user conditions.
2. visit hf.co/pyannote/segmentation and accept user conditions.
3. visit hf.co/settings/tokens to create an access token.

### OpenAI
You also need an organization and API key for OpenAI.

1. visit https://openai.com/blog/openai-api and sign-up.
2. visit https://platform.openai.com/account/org-settings to see your Organization ID.
3. visit https://platform.openai.com/account/api-keys to create a new secret key (API key).


### Run `speech2note`
The command `speech2note` transcribe speech to text using `Whisper`, identify speakers by `pyannote.audio`, and generate text from the results with `OpenAI Chat API`.

```bash
# PDM
$ pdm run speech2note -h

# non-PDM
$ speech2note -h
usage: speech2note [-h] -c CONFIG [--outdir OUTDIR] [--save_transcribe SAVE_TRANSCRIBE] [--save_summary SAVE_SUMMARY]
                   [--skip_diarization] [--skip_chat] [--chunk_increment CHUNK_INCREMENT] [--chunk_size CHUNK_SIZE]
                   audio [audio ...]

positional arguments:
  audio                 audio file to transcribe

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configuratin file (.toml)
  --outdir OUTDIR       Output directory
  --save_transcribe SAVE_TRANSCRIBE
                        Save the diarized transcription by pyannote and whisper. (default: True)
  --save_summary SAVE_SUMMARY
                        Save the summary text by GPT. (default: True)
  --skip_diarization    when set, skip diarization and only make a summary from a CSV file generated before via the
                        script.
  --skip_chat           when set, skip chat with GPT-3.5, i.e., only do diarization by pyannote and whisper.
  --chunk_increment CHUNK_INCREMENT
                        Amount of increment in the transcript for splitting into chunks. (default: 10)
  --chunk_size CHUNK_SIZE
                        Size of chunks in terms of the number of tokens for OpenAI API. (3000 looks ok for English,
                        may need to be <~1000 for Japanese) (default: 3000)
```

Currently, all audio files will be converted to `.wav` file (maybe not needed) and saved in `outdir`.

### Configuration file

A configuration file is also required as a TOML file. An example is the following.

```toml
title = "Configuration file for pyannote-whisper-chatgpt"

[pyannote]
token = "hf_**********"

[pyannote.diarization]
# A negative value means None for "num_speakers", "min_speakers", and "max_speakers"
# "num_speakers" and ("min_speakers" and "max_speakers") are exclusive.
num_speakers = -1
min_speakers = -1
max_speakers = -1


# https://github.com/openai/whisper
[whisper]
# Available models are the following:
# - "tiny"
# - "base"
# - "small"
# - "medium"
# - "large"
# - "tiny.en"
# - "base.en"
# - "small.en"
# - "medium.en"
# - "large.en"

[whisper.model]
name = "base.en"
# device should be one of the following:
# - "cpu"
# - "cuda" (NVIDIA GPU)
# - "mps" (Apple Silicon, but not supported with whisper v20230314)
device = "cpu"

[whisper.transcribe]
language = "en"

# https://platform.openai.com/docs/api-reference/chat
[openai]
model = "gpt-3.5-turbo"
max_tokens = 512
temperature = 0.3
# The prompts are supposed to be used to summarize discussion in a meeting.
prompt = """
Please summarize the following conversation in a meeting and generate the meeting minute. The conversation is a transcription from an audio file, so sometimes you need to guess the context. The conversation is in a CSV format where the first, second, third, and fourth columns indicate the start time, end time, speaker, and text, respectively. When making a minute, please keep the original speaker IDs and clarify who said what. The output must be itemized.
"""
prompt_total = """
Please read the following meeting minutes and generate the grand summary of the minutes. Note that the text is combined one for a long meeting note which is split into chunks so that each part can be summarized by using gpt-3.5-turbo. When making a summary, please keep the original speaker IDs clarify who said what. The output must be itemized.
"""
organization = "org-**********"
api_key = "sk-*********"
```

Please put your token for Hugging Face and Organization ID and API Key for OpenAI accordingly. Do not share your tokens to public.

#### [pyannote] section
You can specify either the number of speakers by `num_speakers` or min/max of the number of speakers by `min_speakers`/`max_speakers`.

#### [whisper] section
`name` can be one of `tiny`, `base`, `small`, `medium`, and `large` (using `.en` can specify English-only models). Using a larger models increase the accuracy with the cost of computational time. For English conversations, `base` seems to work nicely, while larger models would be preferred for Japanese conversations from my initial experiments.

If you have NVIDIA GPUs, you can specify `device = "cuda"` (not tested by myself).

#### [openai] section
`model` can be either `gpt-3.5-turbo` or `gpt-4`. If you have an access to GPT-4 API, you would have better results.

`max_tokens` is used to set a maximum size of tokens returned by the Chat API. The default number (16?) seems small, actually. For English, ~100 may be okay, while Japanese texts use more tokens. Larger number of tokens costs more. You can see the pricing here (https://openai.com/pricing).

`temperature` parameter control the randomness of the answer from the Chat API. Lower temperature reduces the randomness. For meeting notes, lower temperature would be preferred, while higher temperature would suit for more creative text generation.

There are two prompts to the Chat API: `prompt` is used to process the transcribed texts, while `prompt_total` is used to process the answers generated by using the `prompt`.  As an example, suppose you are going to make the minutes for 1 hour long meeting.  The total token for entire transcribed text is order of 10k which exceeds the limit of 4097 tokens for the OpenAI API. In this case, the `speech2note` splits the conversation into small chunks with the size close to `chunk_size` and ask the Chat API to process each of them separately. Then all answers are combined and sent to the Chat API again with the prompt `prompt_total`.


## Example

Let's try to use `speech2note` for an audio file. Here I'll use a audio file in Nature Podcast. You can download and see the transcript at https://www.nature.com/articles/d41586-023-00348-y.

The configuration file is the same as before and saved as `config_base_en_minute.toml`. The command should look like the following.

```
$ mkdir examples
$ cd examples
$ wget https://media.nature.com/original/magazine-assets/d41586-023-00348-y/d41586-023-00348-y_23987834.mpga
# mpga seems to be mp3, so rename it (not quite sure, though)
$ cp d41586-023-00348-y_23987834.mpga d41586-023-00348-y_23987834.mp3

# for PDM
$ pdm run speech2note d41586-023-00348-y_23987834.mp3 -c config_base_en_minute.toml --outdir output

# for others
$ speech2note d41586-023-00348-y_23987834.mp3 -c config_base_en_minute.toml --outdir output
```

I hope that the processing finished without errors. Actually, I once had an error, then I supplied the `.wav` file generated and it went through without problems. I don't know why, maybe there is a bug in the code. I'll look into it. Anyway, it took about 10 minutes on my M1 Pro MacBook Pro, and the following output files are stored in `output/`.  The entire conversation was split into 2 chunks and summarized per chunk. Then summaries for two chunks are merged and summarized by the Chat API again to generate the grand summary.

- `d41586-023-00348-y_23987834.wav` : Audio file converted to `.wav`.
- `d41586-023-00348-y_23987834_transcribed.csv` : Transcribed conversation with timestamps and speaker IDs after merged.
- `d41586-023-00348-y_23987834_summarized_chunks.txt` : Summarized text for each chunks.
- `d41586-023-00348-y_23987834_summarized.txt` : Grand summary of summaries of the chunks.

## References

- [pyannote-whisper](https://github.com/yinruiqing/pyannote-whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
- [Whisper](https://github.com/openai/whisper)
- [OpenAI Documentation](https://platform.openai.com/docs/introduction)
- [OpenAI Whisper tutorial: Whisper - Transcription and diarization (speaker identification)](https://lablab.ai/t/whisper-transcription-and-speaker-identification)
- [ChatGPT APIとFaissを使って長い文章から質問応答する仕組みを作ってみる](https://qiita.com/sakasegawa/items/16714fa132e874cab069)

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Masato Onodera - masato.onodera@gmail.com

## TODO
- https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb

