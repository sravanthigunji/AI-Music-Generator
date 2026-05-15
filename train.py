import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

# Unique notes
unique_notes = sorted(set(notes))

# Mapping
note_to_int = {note: number for number, note in enumerate(unique_notes)}

sequence_length = 50

network_input = []
network_output = []

# Create sequences
for i in range(len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append([note_to_int[note] for note in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

# Reshape input
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# Normalize
network_input = network_input / float(len(unique_notes))

# One-hot output
network_output = to_categorical(network_output)

# Build model
model = Sequential()

model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256))
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(Dense(len(unique_notes), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train model
model.fit(network_input, network_output, epochs=20, batch_size=64)

# Save model
model.save("music_model.h5")

print("Model trained successfully!")