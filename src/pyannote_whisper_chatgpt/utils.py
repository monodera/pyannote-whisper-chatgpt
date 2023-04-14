#!/usr/bin/env python3

import copy
import json
import os
import sys

import openai
import pandas as pd
import tiktoken
import whisper
from logzero import logger
from pyannote.audio import Pipeline
from pydub import AudioSegment

if sys.version_info >= (3, 11):
    try:
        import tomllib
    except ImportError:
        # Help users on older alphas
        if not TYPE_CHECKING:
            import tomli as tomllib
else:
    import tomli as tomllib


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_config(conffile, length_secret=10):
    logger.info(f"Loading and processing config file: {conffile}")
    with open(conffile, mode="rb") as fp:
        config = tomllib.load(fp)

    for k in ["num_speakers", "min_speakers", "max_speakers"]:
        if k in config["pyannote"]["diarization"]:
            if config["pyannote"]["diarization"][k] < 1:
                config["pyannote"]["diarization"][k] = None

    config_masked = copy.deepcopy(config)
    config_masked["pyannote"]["token"] = "*" * length_secret
    config_masked["openai"]["organization"] = "*" * length_secret
    config_masked["openai"]["api_key"] = "*" * length_secret

    logger.info(f"Configuration dictionary {json.dumps(config_masked ,indent=2)}")

    return config


def convert_audio_to_wav(audio_file: str, outdir: str) -> str:
    file_path, ext = os.path.splitext(audio_file)
    ext = ext.replace(".", "")

    if ext in ["wav"]:
        logger.info("Use the original file")
        return audio_file
    else:
        logger.info("Convert the input file to wav file")
        outfile = os.path.join(outdir, f"{os.path.basename(file_path)}.wav")
        logger.info(f"{audio_file} --> {outfile}")
        sound = AudioSegment.from_file(audio_file, format=ext)
        sound.export(outfile, format="wav")
        return outfile


def transcribe_with_whisper(audio, config=None):
    model = whisper.load_model(**config["whisper"]["model"])
    asr_result = model.transcribe(audio, **config["whisper"]["transcribe"])

    return asr_result


def diarization_with_pyannote(audio, config=None):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=config["pyannote"]["token"],
    )
    diarization_result = pipeline(audio, **config["pyannote"]["diarization"])

    return diarization_result


def organize_results(res):
    t_start = []
    t_end = []
    speakers = []
    texts = []

    for seg, spk, sent in res:
        logger.info(f"{seg.start:.2f} {seg.end:.2f} {spk} {sent}")
        t_start.append(seg.start)
        t_end.append(seg.end)
        speakers.append(spk)
        texts.append(sent)

    df = pd.DataFrame(
        {
            "start": t_start,
            "end": t_end,
            "speaker": speakers,
            "text": texts,
        }
    )

    return df


def split_transcription(
    df, prompt_head, chunk_size=3000, encoding="cl100k_base", increment=10
):
    prompt_chunks = []
    df_chunks = []

    idx_start = 0
    idx_end = (
        idx_start + increment
        if (idx_start + increment < df.index.size)
        else (df.index.size)
    )
    tmp_df = df.iloc[idx_start:idx_end]
    tmp_prompt = f"""{prompt_head}\n{tmp_df.to_csv(None, index=False)}"""
    num_tokens = num_tokens_from_string(tmp_prompt, encoding)

    while idx_end <= df.index.size:
        if idx_end == df.index.size:
            logger.info(
                f"Splitting into chunks: Number of tokens {num_tokens} of the last chunk."
            )
            logger.info("Splitting into chunks: Done!")
            prompt_chunks.append(tmp_prompt)
            df_chunks.append(tmp_df)
            break

        # print(num_tokens)
        if num_tokens < chunk_size:
            # print(f"Number of tokens, {num_tokens}, is not enough. increment.")
            idx_end = (
                idx_end + increment
                if (idx_end + increment < df.index.size)
                else (df.index.size)
            )
            tmp_df = df.iloc[idx_start:idx_end]
            tmp_prompt = f"""{prompt_head}\n{tmp_df.to_csv(None, index=False)}"""
            num_tokens = num_tokens_from_string(tmp_prompt, encoding)
        else:
            logger.info(
                f"Splitting into chunks: Number of tokens, {num_tokens}, just exceed the target number {chunk_size}"
            )

            # print(idx_start, idx_end, df.index.size, num_tokens)

            prompt_chunks.append(tmp_prompt)
            df_chunks.append(tmp_df)

            idx_start = idx_end
            idx_end = (
                idx_end + increment
                if (idx_end + increment < df.index.size)
                else (df.index.size)
            )
            tmp_df = df.iloc[idx_start:idx_end]
            tmp_prompt = f"""{prompt_head}\n{tmp_df.to_csv(None, index=False)}"""
            num_tokens = num_tokens_from_string(tmp_prompt, encoding)

    # for d in df_chunks:
    #     print(d.iloc[[0, -1]])

    return prompt_chunks


def summarize_text(df, config=None, increment=10, chunk_size=30):
    openai.organization = config["openai"]["organization"]
    openai.api_key = config["openai"]["api_key"]

    prompt = f"""{config['openai']['prompt']}\n{df.to_csv(None, index=False)}"""

    encoding = tiktoken.encoding_for_model(config["openai"]["model"])

    # print(encoding)

    num_tokens = num_tokens_from_string(prompt, encoding.name)

    # print(prompt)
    logger.info(f"Number of tokens w/o splitting: {num_tokens}")

    # Reference: https://qiita.com/sakasegawa/items/16714fa132e874cab069
    prompt_chunks = split_transcription(
        df,
        config["openai"]["prompt"],
        chunk_size=chunk_size,
        encoding=encoding.name,
        increment=increment,
    )

    summarized_chunks = []

    for prompt in prompt_chunks:
        response_chunks = openai.ChatCompletion.create(
            model=config["openai"]["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional assistant for academics.",
                },
                {
                    "role": "user",
                    "content": f"""{prompt}""",
                },
            ],
            max_tokens=config["openai"]["max_tokens"],
            temperature=config["openai"]["temperature"],
        )
        # print(response_chunks["choices"][0]["message"]["content"])
        logger.info(
            f"Token statistics: {json.dumps(response_chunks['usage'], indent=2)}"
        )
        logger.info(f"Finish reason: {response_chunks['choices'][0]['finish_reason']}")

        summarized_chunks.append(response_chunks["choices"][0]["message"]["content"])

    if len(summarized_chunks) == 1:
        return response_chunks, None

    prompt_total = " ".join(summarized_chunks)
    prompt = f"""{config['openai']['prompt_total']}\n{prompt_total}"""

    response_total = openai.ChatCompletion.create(
        model=config["openai"]["model"],
        messages=[
            {
                "role": "system",
                "content": "You are a professional assistant for academics.",
            },
            {
                "role": "user",
                "content": f"""{prompt}""",
            },
        ],
        max_tokens=config["openai"]["max_tokens"],
    )

    return response_total, summarized_chunks
