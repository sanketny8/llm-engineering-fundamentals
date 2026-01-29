from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from llm_engineering_fundamentals.tokenization.bpe import BPEModel


st.set_page_config(page_title="Token Visualizer (BPE)", layout="wide")
st.title("Token Visualizer (byte-level BPE)")

DEFAULT_MODEL_PATH = "01-tokenization/benchmarks/tiny_bpe.json"

with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    st.caption("Tip: train a model first with `python 01-tokenization/bpe_from_scratch.py --train`")

    st.header("Input")
    text = st.text_area("Text", value="hello tokenization world", height=120)


def load_model(path: str) -> BPEModel:
    data = json.loads(Path(path).read_text())
    return BPEModel.from_json(data)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Token IDs")
    try:
        model = load_model(model_path)
        ids = model.encode(text)
        st.code(ids)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()

with col2:
    st.subheader("Decoded (round-trip)")
    st.code(model.decode(ids))

st.subheader("Symbol view (first 200 symbols)")
syms = [model.id_to_symbol[i] for i in ids[:200]]
st.write(syms)


