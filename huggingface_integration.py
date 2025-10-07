#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face Integration for WiFi Signal-to-Image Conversion
Advanced AI models for interpreting WiFi signals as visual representations
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json
import time
from dataclasses import dataclass

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoImageProcessor,
        ViTModel, ViTImageProcessor,
        ConvNextModel, ConvNextImageProcessor,
        pipeline
    )
    from diffusers import (
        StableDiffusionPipeline, 
        DDIMPipeline,
        LDMPipeline
    )
    import librosa
    import soundfile as sf
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: Hugging Face libraries not available. Install with:")
    print("pip install transformers diffusers librosa soundfile torch torchvision")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Plotting libraries not available.")

@dataclass
class SignalFeatures:
    """Structured signal features for ML models"""
    time_series: np.ndarray
    frequency_spectrum: np.ndarray
    spectrogram: np.ndarray
    mfcc_features: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    chroma_features: np.ndarray
    temporal_features: Dict[str, float]
    statistical_features: Dict[str, float]

class HuggingFaceSignalProcessor:
    """Advanced signal processing using Hugging Face models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.pipelines = {}
        self.load_available_models()
        
    def load_available_models(self):
        """Load available Hugging Face models for signal processing"""
        if not HF_AVAILABLE:
            print("Hugging Face not available - using fallback methods")
            return
            
        try:
            # Vision Transformer for feature extraction
            print("Loading Vision Transformer...")
            self.models['vit'] = ViTModel.from_pretrained('google/vit-base-patch16-224')
            self.models['vit_processor'] = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            
            # Audio processing pipeline
            print("Loading audio processing pipeline...")
            self.pipelines['audio_classification'] = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Image generation pipeline
            print("Loading image generation pipeline...")
            if torch.cuda.is_available():
                self.pipelines['image_generation'] = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                print("GPU not available - skipping image generation pipeline")
            
            # Text-to-image for signal description
            print("Loading text-to-image pipeline...")
            self.pipelines['txt2img'] = pipeline(
                "text-to-image",
                model="CompVis/stable-diffusion-v1-4",
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Continuing with available models...")
    
    def wifi_signal_to_audio(self, signal_data: List, sample_rate: int = 44100) -> np.ndarray:
        """Convert WiFi signal variations to audio representation"""
        if not signal_data:
            return np.array([])
        
        # Extract signal strength time series
        timestamps = [point.timestamp for point in signal_data]
        signal_strengths = [point.signal_strength for point in signal_data]
        frequencies = [point.frequency for point in signal_data]
        
        # Normalize timestamps to audio duration
        if len(timestamps) > 1:
            duration = 5.0  # 5 seconds of audio
            time_normalized = np.linspace(0, duration, len(timestamps))
            
            # Interpolate to audio sample rate
            audio_time = np.linspace(0, duration, int(duration * sample_rate))
            
            # Map signal strength to amplitude (-100 to 0 dBm -> 0 to 1 amplitude)
            amplitudes = [(s + 100) / 100 for s in signal_strengths]
            audio_amplitudes = np.interp(audio_time, time_normalized, amplitudes)
            
            # Map frequencies to audio frequencies (2.4GHz -> 2.4kHz, 5GHz -> 5kHz)
            audio_freqs = [f / 1000000 * 1000 for f in frequencies]  # MHz to kHz
            audio_freq_interp = np.interp(audio_time, time_normalized, audio_freqs)
            
            # Generate audio signal
            audio_signal = np.zeros_like(audio_time)
            for i, (amp, freq) in enumerate(zip(audio_amplitudes, audio_freq_interp)):
                audio_signal[i] = amp * np.sin(2 * np.pi * freq * audio_time[i])
            
            # Add some harmonics for richer sound
            for harmonic in [2, 3, 4]:
                harmonic_signal = np.zeros_like(audio_time)
                for i, (amp, freq) in enumerate(zip(audio_amplitudes, audio_freq_interp)):
                    harmonic_signal[i] = (amp/harmonic) * np.sin(2 * np.pi * freq * harmonic * audio_time[i])
                audio_signal += harmonic_signal
            
            # Normalize
            if np.max(np.abs(audio_signal)) > 0:
                audio_signal = audio_signal / np.max(np.abs(audio_signal))
            
            return audio_signal
        
        return np.array([])
    
    def extract_audio_features(self, audio_signal: np.ndarray, sample_rate: int = 44100) -> SignalFeatures:
        """Extract comprehensive audio features from signal"""
        if len(audio_signal) == 0:
            # Return empty features
            return SignalFeatures(
                time_series=np.array([]),
                frequency_spectrum=np.array([]),
                spectrogram=np.array([[]]),
                mfcc_features=np.array([[]]),
                spectral_centroid=np.array([]),
                spectral_rolloff=np.array([]),
                zero_crossing_rate=np.array([]),
                chroma_features=np.array([[]]),
                temporal_features={},
                statistical_features={}
            )
        
        try:
            # Time series features
            time_series = audio_signal
            
            # Frequency spectrum (FFT)
            frequency_spectrum = np.abs(np.fft.fft(audio_signal))
            
            # Spectrogram
            spectrogram = librosa.stft(audio_signal, hop_length=512)
            spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))
            
            # MFCC features
            mfcc_features = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sample_rate)
            
            # Zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_signal)
            
            # Chroma features
            chroma_features = librosa.feature.chroma_stft(y=audio_signal, sr=sample_rate)
            
            # Temporal features
            temporal_features = {
                'duration': len(audio_signal) / sample_rate,
                'rms_energy': float(np.sqrt(np.mean(audio_signal**2))),
                'max_amplitude': float(np.max(np.abs(audio_signal))),
                'min_amplitude': float(np.min(np.abs(audio_signal))),
            }
            
            # Statistical features
            statistical_features = {
                'mean': float(np.mean(audio_signal)),
                'std': float(np.std(audio_signal)),
                'skewness': float(self._calculate_skewness(audio_signal)),
                'kurtosis': float(self._calculate_kurtosis(audio_signal)),
                'energy': float(np.sum(audio_signal**2)),
            }
            
            return SignalFeatures(
                time_series=time_series,
                frequency_spectrum=frequency_spectrum,
                spectrogram=spectrogram_db,
                mfcc_features=mfcc_features,
                spectral_centroid=spectral_centroid[0],
                spectral_rolloff=spectral_rolloff[0],
                zero_crossing_rate=zero_crossing_rate[0],
                chroma_features=chroma_features,
                temporal_features=temporal_features,
                statistical_features=statistical_features
            )
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return SignalFeatures(
                time_series=audio_signal,
                frequency_spectrum=np.array([]),
                spectrogram=np.array([[]]),
                mfcc_features=np.array([[]]),
                spectral_centroid=np.array([]),
                spectral_rolloff=np.array([]),
                zero_crossing_rate=np.array([]),
                chroma_features=np.array([[]]),
                temporal_features={},
                statistical_features={}
            )
    
    def generate_signal_description(self, features: SignalFeatures) -> str:
        """Generate textual description of signal characteristics"""
        description_parts = []
        
        # Analyze temporal characteristics
        if features.temporal_features:
            duration = features.temporal_features.get('duration', 0)
            energy = features.temporal_features.get('rms_energy', 0)
            
            if duration > 3:
                description_parts.append("long duration WiFi signal")
            elif duration > 1:
                description_parts.append("medium duration WiFi signal")
            else:
                description_parts.append("short burst WiFi signal")
            
            if energy > 0.5:
                description_parts.append("with high signal strength")
            elif energy > 0.2:
                description_parts.append("with moderate signal strength")
            else:
                description_parts.append("with weak signal strength")
        
        # Analyze spectral characteristics
        if len(features.spectral_centroid) > 0:
            avg_centroid = np.mean(features.spectral_centroid)
            if avg_centroid > 3000:
                description_parts.append("high frequency content")
            elif avg_centroid > 1000:
                description_parts.append("mixed frequency content")
            else:
                description_parts.append("low frequency content")
        
        # Analyze statistical properties
        if features.statistical_features:
            std = features.statistical_features.get('std', 0)
            if std > 0.3:
                description_parts.append("highly variable signal")
            elif std > 0.1:
                description_parts.append("moderately variable signal")
            else:
                description_parts.append("stable signal")
        
        # Interference indicators
        if len(features.zero_crossing_rate) > 0:
            avg_zcr = np.mean(features.zero_crossing_rate)
            if avg_zcr > 0.3:
                description_parts.append("with potential interference patterns")
        
        # Create final description
        if description_parts:
            base_description = ", ".join(description_parts[:3])  # Limit length
            
            # Add WiFi-specific context
            wifi_context = "showing electromagnetic field variations in 2.4GHz and 5GHz bands"
            
            full_description = f"Visualization of {base_description}, {wifi_context}, rendered as colorful heatmap with signal propagation patterns"
        else:
            full_description = "WiFi signal visualization showing electromagnetic field strength patterns with interference analysis"
        
        return full_description
    
    def features_to_image_with_ai(self, features: SignalFeatures) -> Optional[np.ndarray]:
        """Use AI models to convert features to image"""
        if not HF_AVAILABLE or 'txt2img' not in self.pipelines:
            print("AI image generation not available, using fallback method")
            return self._features_to_image_fallback(features)
        
        try:
            # Generate description
            description = self.generate_signal_description(features)
            
            print(f"Generating image from description: {description}")
            
            # Generate image using text-to-image pipeline
            generated = self.pipelines['txt2img'](
                description,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            )
            
            if hasattr(generated, 'images') and len(generated.images) > 0:
                # Convert PIL image to numpy array
                pil_image = generated.images[0]
                numpy_image = np.array(pil_image)
                
                # Convert RGB to BGR for OpenCV
                if len(numpy_image.shape) == 3:
                    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                
                return numpy_image
            
        except Exception as e:
            print(f"AI image generation failed: {e}")
            print("Falling back to traditional method")
        
        return self._features_to_image_fallback(features)
    
    def _features_to_image_fallback(self, features: SignalFeatures) -> np.ndarray:
        """Fallback method to convert features to image"""
        # Create image from spectrogram
        if features.spectrogram.shape[0] > 0 and features.spectrogram.shape[1] > 0:
            # Resize spectrogram to reasonable size
            target_size = (512, 512)
            spectrogram_resized = cv2.resize(
                features.spectrogram, 
                target_size, 
                interpolation=cv2.INTER_CUBIC
            )
            
            # Normalize to 0-255 range
            if np.max(spectrogram_resized) > np.min(spectrogram_resized):
                normalized = ((spectrogram_resized - np.min(spectrogram_resized)) / 
                             (np.max(spectrogram_resized) - np.min(spectrogram_resized)) * 255).astype(np.uint8)
            else:
                normalized = np.zeros(target_size, dtype=np.uint8)
            
            # Apply colormap
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_PLASMA)
            
            return colored
        
        # If no spectrogram, create from frequency spectrum
        elif len(features.frequency_spectrum) > 0:
            # Create 2D representation from 1D spectrum
            spectrum = features.frequency_spectrum[:512]  # Limit size
            
            # Create circular pattern
            image_size = 512
            center = image_size // 2
            image = np.zeros((image_size, image_size), dtype=np.float32)
            
            for i, magnitude in enumerate(spectrum):
                radius = int(i / len(spectrum) * center)
                if radius < center:
                    # Create circle at radius with magnitude intensity
                    y, x = np.ogrid[:image_size, :image_size]
                    mask = ((x - center)**2 + (y - center)**2) >= radius**2
                    mask &= ((x - center)**2 + (y - center)**2) < (radius + 2)**2
                    image[mask] = magnitude
            
            # Normalize and apply colormap
            if np.max(image) > 0:
                normalized = (image / np.max(image) * 255).astype(np.uint8)
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
                return colored
        
        # Ultimate fallback - create pattern from time series
        if len(features.time_series) > 0:
            # Create simple waveform visualization
            image_size = (512, 512)
            image = np.zeros(image_size, dtype=np.uint8)
            
            # Downsample time series to fit image width
            time_series = features.time_series
            if len(time_series) > image_size[1]:
                indices = np.linspace(0, len(time_series)-1, image_size[1]).astype(int)
                time_series = time_series[indices]
            
            # Map amplitudes to y coordinates
            normalized_amplitudes = ((time_series + 1) / 2 * (image_size[0] - 1)).astype(int)
            
            # Draw waveform
            for x, y in enumerate(normalized_amplitudes):
                if 0 <= y < image_size[0]:
                    image[y, x] = 255
            
            # Apply colormap
            colored = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
            return colored
        
        # Final fallback - return solid color
        return np.full((512, 512, 3), (128, 128, 128), dtype=np.uint8)
    
    def analyze_with_vision_model(self, image: np.ndarray) -> Dict:
        """Analyze generated image using vision transformer"""
        if not HF_AVAILABLE or 'vit' not in self.models:
            return {"analysis": "Vision model not available"}
        
        try:
            # Prepare image for ViT
            if len(image.shape) == 3:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Preprocess
            inputs = self.models['vit_processor'](
                images=image_rgb, 
                return_tensors="pt"
            )
            
            # Get features
            with torch.no_grad():
                outputs = self.models['vit'](inputs.pixel_values.to(self.device))
                features = outputs.last_hidden_state
            
            # Analyze features
            feature_stats = {
                "mean_activation": float(torch.mean(features).cpu()),
                "max_activation": float(torch.max(features).cpu()),
                "min_activation": float(torch.min(features).cpu()),
                "feature_variance": float(torch.var(features).cpu()),
                "num_features": features.shape[-1],
                "spatial_dimensions": (features.shape[1], features.shape[2]) if len(features.shape) > 2 else (1, 1)
            }
            
            return {
                "analysis": "ViT analysis completed",
                "feature_statistics": feature_stats,
                "interpretation": self._interpret_vision_features(feature_stats)
            }
            
        except Exception as e:
            return {"analysis": f"Vision analysis failed: {str(e)}"}
    
    def _interpret_vision_features(self, stats: Dict) -> str:
        """Interpret vision model features"""
        interpretation_parts = []
        
        mean_activation = stats.get("mean_activation", 0)
        variance = stats.get("feature_variance", 0)
        
        if mean_activation > 0.5:
            interpretation_parts.append("strong visual patterns detected")
        elif mean_activation > 0.1:
            interpretation_parts.append("moderate visual complexity")
        else:
            interpretation_parts.append("subtle visual features")
        
        if variance > 1.0:
            interpretation_parts.append("high pattern diversity")
        elif variance > 0.1:
            interpretation_parts.append("moderate pattern variation")
        else:
            interpretation_parts.append("uniform pattern distribution")
        
        return ", ".join(interpretation_parts)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def save_audio_representation(self, audio_signal: np.ndarray, filename: str, sample_rate: int = 44100):
        """Save audio representation of WiFi signals"""
        try:
            sf.write(filename, audio_signal, sample_rate)
            print(f"Audio representation saved to: {filename}")
        except Exception as e:
            print(f"Failed to save audio: {e}")

class WiFiSignalToImageAI:
    """Main class for AI-powered WiFi signal to image conversion"""
    
    def __init__(self):
        self.hf_processor = HuggingFaceSignalProcessor()
        self.output_dir = Path("wifi_ai_generated")
        self.output_dir.mkdir(exist_ok=True)
        
    def process_wifi_signals(self, signal_data: List, output_prefix: str = "wifi_signal") -> Dict:
        """Complete pipeline: WiFi signals -> Audio -> Features -> AI Image"""
        results = {
            "timestamp": time.time(),
            "input_signals": len(signal_data),
            "processing_stages": {},
            "output_files": []
        }
        
        try:
            print("Step 1: Converting WiFi signals to audio representation...")
            audio_signal = self.hf_processor.wifi_signal_to_audio(signal_data)
            results["processing_stages"]["audio_conversion"] = {
                "success": True,
                "audio_length": len(audio_signal),
                "duration_seconds": len(audio_signal) / 44100 if len(audio_signal) > 0 else 0
            }
            
            # Save audio representation
            if len(audio_signal) > 0:
                audio_filename = self.output_dir / f"{output_prefix}_audio.wav"
                self.hf_processor.save_audio_representation(audio_signal, str(audio_filename))
                results["output_files"].append(str(audio_filename))
            
            print("Step 2: Extracting comprehensive audio features...")
            features = self.hf_processor.extract_audio_features(audio_signal)
            results["processing_stages"]["feature_extraction"] = {
                "success": True,
                "features_extracted": {
                    "time_series_length": len(features.time_series),
                    "frequency_bins": len(features.frequency_spectrum),
                    "spectrogram_shape": features.spectrogram.shape,
                    "mfcc_coefficients": features.mfcc_features.shape[0] if len(features.mfcc_features.shape) > 0 else 0
                }
            }
            
            print("Step 3: Generating AI-powered image visualization...")
            ai_image = self.hf_processor.features_to_image_with_ai(features)
            
            if ai_image is not None:
                # Save AI-generated image
                ai_image_filename = self.output_dir / f"{output_prefix}_ai_generated.png"
                cv2.imwrite(str(ai_image_filename), ai_image)
                results["output_files"].append(str(ai_image_filename))
                
                results["processing_stages"]["ai_image_generation"] = {
                    "success": True,
                    "image_shape": ai_image.shape,
                    "image_size_mb": os.path.getsize(ai_image_filename) / (1024 * 1024)
                }
                
                print("Step 4: Analyzing generated image with Vision Transformer...")
                vision_analysis = self.hf_processor.analyze_with_vision_model(ai_image)
                results["processing_stages"]["vision_analysis"] = vision_analysis
                
                # Save traditional visualizations for comparison
                print("Step 5: Creating additional visualizations...")
                self._create_traditional_visualizations(features, output_prefix, results)
                
            else:
                results["processing_stages"]["ai_image_generation"] = {
                    "success": False,
                    "error": "Failed to generate AI image"
                }
            
        except Exception as e:
            results["processing_stages"]["error"] = {
                "stage": "unknown",
                "error": str(e)
            }
            print(f"Error in processing pipeline: {e}")
        
        # Save processing report
        report_filename = self.output_dir / f"{output_prefix}_processing_report.json"
        with open(report_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        results["output_files"].append(str(report_filename))
        
        return results
    
    def _create_traditional_visualizations(self, features: SignalFeatures, output_prefix: str, results: Dict):
        """Create traditional visualizations for comparison"""
        try:
            if PLOTTING_AVAILABLE:
                # Spectrogram plot
                if features.spectrogram.shape[0] > 0:
                    plt.figure(figsize=(10, 6))
                    librosa.display.specshow(
                        features.spectrogram, 
                        sr=44100, 
                        hop_length=512,
                        x_axis='time', 
                        y_axis='hz'
                    )
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('WiFi Signal Spectrogram')
                    plt.tight_layout()
                    
                    spec_filename = self.output_dir / f"{output_prefix}_spectrogram.png"
                    plt.savefig(spec_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    results["output_files"].append(str(spec_filename))
                
                # MFCC visualization
                if features.mfcc_features.shape[0] > 0:
                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(
                        features.mfcc_features,
                        sr=44100,
                        x_axis='time'
                    )
                    plt.colorbar()
                    plt.title('MFCC Features from WiFi Signal')
                    plt.tight_layout()
                    
                    mfcc_filename = self.output_dir / f"{output_prefix}_mfcc.png"
                    plt.savefig(mfcc_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    results["output_files"].append(str(mfcc_filename))
                
                # Feature summary plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('WiFi Signal Analysis Summary')
                
                # Time series
                if len(features.time_series) > 0:
                    axes[0, 0].plot(features.time_series[:1000])  # Limit for visibility
                    axes[0, 0].set_title('Time Series (First 1000 samples)')
                    axes[0, 0].set_xlabel('Sample')
                    axes[0, 0].set_ylabel('Amplitude')
                
                # Frequency spectrum
                if len(features.frequency_spectrum) > 0:
                    freqs = np.fft.fftfreq(len(features.frequency_spectrum), 1/44100)
                    axes[0, 1].plot(freqs[:len(freqs)//2], features.frequency_spectrum[:len(features.frequency_spectrum)//2])
                    axes[0, 1].set_title('Frequency Spectrum')
                    axes[0, 1].set_xlabel('Frequency (Hz)')
                    axes[0, 1].set_ylabel('Magnitude')
                
                # Spectral centroid
                if len(features.spectral_centroid) > 0:
                    axes[1, 0].plot(features.spectral_centroid)
                    axes[1, 0].set_title('Spectral Centroid')
                    axes[1, 0].set_xlabel('Frame')
                    axes[1, 0].set_ylabel('Hz')
                
                # Zero crossing rate
                if len(features.zero_crossing_rate) > 0:
                    axes[1, 1].plot(features.zero_crossing_rate)
                    axes[1, 1].set_title('Zero Crossing Rate')
                    axes[1, 1].set_xlabel('Frame')
                    axes[1, 1].set_ylabel('Rate')
                
                plt.tight_layout()
                summary_filename = self.output_dir / f"{output_prefix}_analysis_summary.png"
                plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
                plt.close()
                results["output_files"].append(str(summary_filename))
            
            print("Traditional visualizations created successfully")
            
        except Exception as e:
            print(f"Error creating traditional visualizations: {e}")

def main():
    """Test the Hugging Face integration"""
    print("WiFi Signal to Image AI Conversion Test")
    print("=" * 50)
    
    # Create mock WiFi signal data
    from wifi_signal_capture import WiFiSignalPoint
    import random
    
    print("Creating mock WiFi signal data...")
    signal_data = []
    current_time = time.time()
    
    for i in range(200):  # 200 signal points
        signal_point = WiFiSignalPoint(
            timestamp=current_time + i * 0.1,
            frequency=2437 + random.choice([-25, 0, 25]),  # Channel variation
            signal_strength=-50 + random.gauss(0, 15),     # Random signal strength
            noise_level=-95 + random.gauss(0, 5),
            channel=random.choice([1, 6, 11]),
            ssid=random.choice(["TestWiFi", "HomeNet", "Office5G"]),
            bssid=f"aa:bb:cc:dd:ee:{i:02x}",
            x=random.uniform(0, 800),
            y=random.uniform(0, 600),
            phase=random.uniform(0, 2*np.pi),
            bandwidth=20
        )
        signal_data.append(signal_point)
    
    print(f"Generated {len(signal_data)} signal points")
    
    # Process with AI
    converter = WiFiSignalToImageAI()
    
    print("Processing WiFi signals with AI models...")
    results = converter.process_wifi_signals(signal_data, "test_wifi_signal")
    
    print("\n" + "="*50)
    print("PROCESSING RESULTS:")
    print("="*50)
    
    for stage, result in results["processing_stages"].items():
        print(f"\n{stage.upper()}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {result}")
    
    print(f"\nOUTPUT FILES GENERATED:")
    for file_path in results["output_files"]:
        print(f"  - {file_path}")
    
    print(f"\nAll files saved to: {converter.output_dir}")
    print("Processing complete!")

if __name__ == "__main__":
    main()