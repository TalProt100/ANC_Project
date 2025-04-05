import streamlit as st
import numpy as np
import wave
import io
import matplotlib.pyplot as plt

###################################
# STEP 1: LMS FILTER
###################################
def lms_filter(noisy_signal, reference_signal, filter_length=32, mu=0.001):
    n_samples = len(noisy_signal)
    w = np.zeros(filter_length)
    filtered_signal = np.zeros(n_samples)

    for n in range(n_samples):
        # Create a sliding window, x_n
        x_n = np.zeros(filter_length)
        for k in range(filter_length):
            if (n - k) >= 0:
                x_n[k] = noisy_signal[n - k]
        y_n = np.dot(w, x_n)
        e_n = reference_signal[n] - y_n
        w += mu * e_n * x_n
        filtered_signal[n] = y_n

    return filtered_signal, w

###################################
# STEP 2: HELPER FUNCTIONS
###################################
def generate_signals(freq=440.0, sample_rate=16000, duration=3.0, noise_amp=0.4,
                     filter_length=64, mu=0.001):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    original = 0.6 * np.sin(2 * np.pi * freq * t)
    noise = noise_amp * np.random.randn(len(t))
    noisy = original + noise
    filtered, _ = lms_filter(noisy, original, filter_length=filter_length, mu=mu)
    return original, noisy, filtered, t

def to_wav_bytes(audio_data, sample_rate=16000):
    """
    Convert a NumPy audio signal to an in-memory WAV file (BytesIO),
    which can then be passed to st.audio().
    """
    # Scale float (-1.0 to 1.0) to int16
    scaled = np.int16(audio_data * 32767)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(scaled.tobytes())
    wav_buf.seek(0)  # Important: reset buffer to the beginning
    return wav_buf

###################################
# STEP 3: STREAMLIT APP
###################################
def main():
    st.title("Noise Cancellation Demo - LMS Filter")
    st.markdown("**by Yuval and Tal**")

    # Sidebar controls
    st.sidebar.header("Parameters")
    freq = st.sidebar.slider("Signal Frequency (Hz)", 50, 2000, 440, step=10)
    mu = st.sidebar.slider("Step Size (μ)", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
    filter_len = st.sidebar.slider("Filter Length", 8, 128, 64, step=1)
    noise_amp = 0.4      # Could also expose as a slider if desired
    sample_rate = 16000  # Could also expose if desired
    duration = 3.0       # seconds

    # Generate signals based on the selected parameters
    original, noisy, filtered, t = generate_signals(
        freq=freq,
        sample_rate=sample_rate,
        duration=duration,
        noise_amp=noise_amp,
        filter_length=filter_len,
        mu=mu
    )

    # Plot waveforms
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("Signal Waveforms")

    axes[0].plot(t, original)
    axes[0].set_title("Original Signal")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, noisy)
    axes[1].set_title("Noisy Signal")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(t, filtered)
    axes[2].set_title(f"Filtered Signal (L={filter_len}, μ={mu})")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")

    st.pyplot(fig)

    # Convert each signal to WAV and offer playback
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.audio(to_wav_bytes(original, sample_rate), format='audio/wav')
    with col2:
        st.subheader("Noisy")
        st.audio(to_wav_bytes(noisy, sample_rate), format='audio/wav')
    with col3:
        st.subheader("Filtered")
        st.audio(to_wav_bytes(filtered, sample_rate), format='audio/wav')

if __name__ == "__main__":
    main()
