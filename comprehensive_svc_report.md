# Comprehensive Report: Singing Voice Conversion System

This report details the functionality, workflow, and usage of the Singing Voice Conversion (SVC) system, based on the provided codebase structure and README file.

**1. Project Functionality Summary**

*   **What it is:** The project is a Singing Voice Conversion (SVC) system. Its primary purpose is to take a source singing audio track (acapella) and transform its vocal characteristics to match those of a target singer. This effectively allows "cloning" a target singer's voice onto a different vocal performance. A key feature is its ability to be fine-tuned for new singers, meaning you can train it on a specific artist's vocals to create a model of their unique voice.
*   **Core Technologies:**
    *   **OpenAI Whisper:** Utilized for content encoding. It processes input audio to extract Phonetic PosteriorGrams (PPGs), which represent the linguistic content of the song (the "what" is being sung), independent of the original singer's vocal timbre.
    *   **Nvidia BigVGAN:** A neural vocoder (a type of Generative Adversarial Network - GAN) responsible for generating high-quality audio output. It synthesizes the new singing voice by taking the content encoding (from Whisper) and the target speaker's timbre information. BigVGAN is known for producing realistic speech and reducing artifacts.
    *   **Microsoft Adapter Concept (LoRA mentioned):** The system employs an adapter-based approach for efficient fine-tuning. This allows a pre-trained model to be adapted to new singers without retraining the entire large model from scratch. While LoRA (Low-Rank Adaptation) is mentioned, the core idea is adapting a base model.
    *   **Neural Source-Filter Model:** The underlying acoustic modeling approach likely decomposes the voice into a source (excitation, related to pitch) and a filter (representing vocal tract resonances, related to timbre), which BigVGAN then synthesizes.

**2. Workflow/User Flow Explanation**

The general process of using this SVC system involves the following main stages:

1.  **Prerequisites & Setup:**
    *   **Acapella Preparation:** Obtain clean acapella (vocals-only) audio. Remove any instrumental accompaniment (e.g., using [UVR](https://github.com/Anjok07/ultimatevocalremovergui)). Segment acapellas into short clips (< 30 seconds) using a tool like [audio-slicer](https://github.com/flutydeer/audio-slicer).
    *   **Directory Structure:** Organize raw audio data: `data_raw/speaker_name/audio_segment.wav`.
    *   **Install Dependencies:** Install Python packages (`pip install -r requirements.txt`). Download pre-trained models: Timbre Encoder (`best_model.pth.tar` into `speaker_pretrain/`), Whisper model (`medium.pt` into `whisper_pretrain/`), and the base SVC model (`maxgan_pretrain_32K.pth`).

2.  **Data Preprocessing:**
    This converts raw audio into features the model uses. An `easyprocess.py` script can automate these, or they can be run manually:
    *   **Resampling:** Create 16kHz (for feature extraction) and 32kHz (for training) versions of audio.
    *   **Pitch (F0) Extraction:** From 16kHz audio (`preprocess_f0.py`).
    *   **PPG (Content) Extraction:** Using Whisper on 16kHz audio (`preprocess_ppg.py`).
    *   **Timbre Code Extraction:** Speaker embeddings from 16kHz audio (`preprocess_speaker.py`).
    *   **Average Singer Code Extraction:** For inference (`preprocess_speaker_ave.py`), creates `data_svc/singer/speaker_name.spk.npy`.
    *   **Training Index Generation:** (`preprocess_train.py`).
    *   The output is a structured `data_svc/` directory.

3.  **Training (Fine-tuning for a New Speaker):**
    Necessary for cloning a voice not already available.
    *   **Configuration:** Edit `configs/maxgan.yaml`. Set `pretrain` to `"./maxgan_pretrain_32K.pth"`. Adjust learning rate (e.g., `1e-5`). Ensure paths point to preprocessed data.
    *   **Start Training:** `python svc_trainer.py -c configs/maxgan.yaml -n your_model_name`.
    *   **Monitor Training:** Use TensorBoard: `tensorboard --logdir logs/`.

4.  **Inference (Generating Voice-Cloned Audio):**
    *   **A. Using the GUI (`svc_gui.py`):**
        *   Launch: `python3 svc_gui.py`.
        *   Select model, target speaker's `.npy` file, and input acapella.
    *   **B. Using Command-Line Interface:**
        1.  **Export Model:** `python svc_export.py --config ... --checkpoint_path ...` (creates e.g., `maxgan_g.pth`).
        2.  **Prepare Input Acapella:** (Optional but recommended) Extract PPG (`whisper/inference.py`) and Pitch (`pitch/inference.py`).
        3.  **Run Inference:** `python svc_inference.py --config ... --model ... --spk ... --wave ... [--ppg ... --pit ...]`. Output is `svc_out.wav`.
        4.  **Post-processing (Optional):** `python svc_inference_post.py ...`.

**3. Detailed Voice Cloning Process for a New Artist**

Here’s how to use the system to clone the singing voice of a new artist using their acapella files:

**Phase 1: Preparation and Setup**
1.  **Gather Artist's Acapellas:** Collect high-quality, vocals-only `.wav` recordings. Use UVR for vocal separation and audio-slicer for segments < 30s.
2.  **Organize Raw Data:** Create `data_raw/YourArtistName/` and place acapella segments there.
3.  **Install System Dependencies:** As described in the general workflow.

**Phase 2: Data Preprocessing for the New Artist**
Use `python3 prepare/easyprocess.py` or run manual steps:
1.  **Resample Audio:** To `data_svc/waves-16k/YourArtistName/` and `data_svc/waves-32k/YourArtistName/`.
2.  **Extract Pitch (F0):** To `data_svc/pitch/YourArtistName/`.
3.  **Extract PPG (Whisper Content):** To `data_svc/whisper/YourArtistName/`.
4.  **Extract Timbre Codes (Per File):** To `data_svc/speaker/YourArtistName/`.
5.  **Extract Average Singer Code:** Creates `data_svc/singer/YourArtistName.spk.npy`.
6.  **Generate Training Index:** `preprocess_train.py` (ensure your artist's data is included).

**Phase 3: Fine-Tuning the Model for the New Artist**
1.  **Configure Training:** Copy `configs/maxgan.yaml` to `configs/maxgan_yourartist.yaml`.
    *   Edit `configs/maxgan_yourartist.yaml`:
        *   Set `pretrain: "./maxgan_pretrain_32K.pth"`.
        *   Set `lr: 1e-5`.
        *   Ensure `training_files` and `validation_files` point to the correct file lists.
2.  **Start Fine-Tuning:** `python svc_trainer.py -c configs/maxgan_yourartist.yaml -n YourArtistSVCName`. Checkpoints save to `chkpt/YourArtistSVCName/`.
3.  **Monitor Training:** `tensorboard --logdir logs/`.

**Phase 4: Inference (Cloning Voice onto a New Song)**
1.  **Export Fine-Tuned Model:** `python svc_export.py --config configs/maxgan_yourartist.yaml --checkpoint_path chkpt/YourArtistSVCName/best_checkpoint.pth`. This creates e.g., `yourartist_g.pth`.
2.  **Prepare Input Song (`source_song.wav`):** This is the acapella you want your cloned artist to "sing."
    *   (Recommended) Extract PPG: `python whisper/inference.py -w source_song.wav -p source_song.ppg.npy`.
    *   (Recommended) Extract Pitch: `python pitch/inference.py -w source_song.wav -p source_song.csv`.
3.  **Run Voice Conversion:**
    ```bash
    python svc_inference.py \
          --config configs/maxgan_yourartist.yaml \
          --model yourartist_g.pth \
          --spk ./data_svc/singer/YourArtistName.spk.npy \
          --wave source_song.wav \
          --ppg source_song.ppg.npy \
          --pit source_song.csv
    ```
    Output is `svc_out.wav`.
4.  **Post-Process (Optional):** `python svc_inference_post.py --ref source_song.wav --svc svc_out.wav --out YourArtist_cloned_final.wav`.

**4. Direct Answers to Specific Questions**

*   **How does this work?**
    It separates the content (what is sung, via Whisper PPGs), pitch (melody), and timbre (vocal quality, via a Speaker Encoder). It then uses a powerful neural synthesizer (BigVGAN) to recombine these elements, allowing the timbre to be swapped to that of a target singer. Fine-tuning adapts a base model to new voices efficiently.

*   **What are all the functions that it can do?**
    *   Singing Voice Conversion (SVC).
    *   New speaker fine-tuning (voice cloning).
    *   Comprehensive data preprocessing (resampling, pitch/PPG/timbre extraction).
    *   Model training and inference (with GUI or CLI options).
    *   Model export for inference.
    *   Optional output post-processing.

*   **What is the user flow and step process for it to work?**
    The flow is:
    1.  **Setup & Installation:** Install dependencies, download pre-trained models.
    2.  **Dataset Preparation:** Collect, clean, and organize acapellas for target singers.
    3.  **Data Preprocessing:** Convert audio to model-usable features (pitch, PPGs, timbre codes).
    4.  **Training (Fine-tuning):** Adapt the model to new voices using their preprocessed data.
    5.  **Inference:** Generate voice-cloned audio using the trained model and an input song.

*   **How can I use it to make singing voice clones of artists I like with acapella files?**
    Follow the "Detailed Voice Cloning Process for a New Artist" in section 3 of this report. This involves gathering the artist's acapellas, preprocessing them, fine-tuning the SVC model on this data, and then using the fine-tuned model to convert new songs into that artist's voice.

This concludes the report on the Singing Voice Conversion system.
