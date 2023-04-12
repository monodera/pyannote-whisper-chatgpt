#!/usr/bin/env python3

import argparse
import json
import os
import time

import pandas as pd
from logzero import logger
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text

from pyannote_whisper_chatgpt.utils import (
    convert_audio_to_wav,
    diarization_with_pyannote,
    load_config,
    organize_results,
    summarize_text,
    transcribe_with_whisper,
)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("audio", nargs="+", type=str, help="audio file to transcribe")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Configuratin file (.toml)"
    )
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")

    parser.add_argument(
        "--save_transcribe",
        type=bool,
        default=True,
        help="Save the diarized transcription by pyannote and whisper. (default: True)",
    )
    parser.add_argument(
        "--save_summary",
        type=bool,
        default=True,
        help="Save the summary text by GPT. (default: True)",
    )
    parser.add_argument(
        "--skip_diarization",
        action="store_true",
        help="when set, skip diarization and only make a summary from a CSV file generated before via the script.",
    )
    parser.add_argument(
        "--skip_chat",
        action="store_true",
        help="when set, skip chat with GPT-3.5, i.e., only do diarization by pyannote and whisper.",
    )
    parser.add_argument(
        "--chunk_increment",
        type=int,
        default=10,
        help="Amount of increment in the transcript for splitting into chunks. (default: 10)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3000,
        help="Size of chunks in terms of the number of tokens for OpenAI API. (3000 looks ok for English, may need to be <~1000 for Japanese) (default: 3000)",
    )

    args = parser.parse_args()

    logger.info(f"Parsing command-line arguments: {json.dumps(vars(args), indent=2)}")

    return args


def speech2note():
    args = get_arguments()

    # Create output directory if not exists
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    config = load_config(args.config)

    if not args.skip_diarization:
        audio_file = convert_audio_to_wav(args.audio[0], args.outdir)
        file_prefix = os.path.basename(os.path.splitext(audio_file)[0])

        # Use Whisper for transcription
        t_begin_whisper = time.time()
        asr_result = transcribe_with_whisper(audio_file, config=config)
        t_end_whisper = time.time()
        logger.info(
            f"Elapsed time for transcription with whisper: {t_end_whisper - t_begin_whisper:.2f}s"
        )

        # Use pyannote.audio for speaker identification
        t_begin_pyannote = time.time()
        diarization_result = diarization_with_pyannote(audio_file, config=config)
        t_end_pyannote = time.time()
        logger.info(
            f"Elapsed time for diarization with pyannote: {t_end_pyannote - t_begin_pyannote:.2f}s"
        )

        # Assign speakers to the transcription
        final_result = diarize_text(asr_result, diarization_result)

        # Put the result in a pandas.DataFrame
        df = organize_results(final_result)

        # Save the transription in a CSV file
        if args.save_transcribe:
            df.to_csv(
                os.path.join(args.outdir, f"{file_prefix}_transcribed.csv"),
                index=False,
            )
    else:
        file_prefix = os.path.basename(os.path.splitext(args.audio[0])[0])
        df = pd.read_csv(os.path.join(args.outdir, f"{file_prefix}_transcribed.csv"))

    if args.skip_chat:
        logger.info("Skip chat with GPT-3.5 and return")
        return

    # Use GPT via OpenAI API to make a summary
    t_begin_chat = time.time()
    response, response_chunks = summarize_text(
        df,
        config=config,
        increment=args.chunk_increment,
        chunk_size=args.chunk_size,
    )
    t_end_chat = time.time()
    logger.info(f"Elapsed time for chat with GPT: {t_end_chat - t_begin_chat:.2f}s")

    # Write the summary to an ASCII file
    logger.info("Writing the results.")
    if args.save_summary:
        with open(
            os.path.join(args.outdir, f"{file_prefix}_summarized.txt"), "w"
        ) as outfile:
            res = response["choices"][0]["message"]["content"]
            logger.info(f"""# final response:\n\n{res}""")
            outfile.write(res)
            outfile.write("\n")

        if response_chunks is not None:
            with open(
                os.path.join(args.outdir, f"{file_prefix}_summarized_chunks.txt"), "w"
            ) as outfile:
                for i, r in enumerate(response_chunks):
                    logger.info(f"""response_chunk #{i}:\n {r}""")
                    outfile.write(f"## Output from chunk {i}\n\n")
                    outfile.write(r)
                    outfile.write("\n\n")


if __name__ == "__main__":
    speech2note()
