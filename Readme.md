# Digit Vision — MNIST CNN Streamlit App

## Files needed in the same folder
- `app.py`
- `requirements.txt`
- `mnist_cnn.keras` ← from your Colab notebook (already saved via `model.save("mnist_cnn.keras")`, just download it and drop it here)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `mnist_cnn.keras` to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set `app.py` as the entry point.
3. Deploy.

## Features
- Draw-a-digit canvas (freehand) with auto centering/cropping to match MNIST preprocessing
- Upload-an-image mode with automatic black/white polarity detection
- Live prediction with confidence bar chart across all 10 digits
- Sidebar with model architecture + accuracy stats
- Custom dark, gradient-accented UI