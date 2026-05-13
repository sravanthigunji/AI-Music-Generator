import pickle
import numpy as np

from tensorflow.keras.models import load_model
from music21 import stream, note, chord, instrument

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

# Unique notes
unique_notes = sorted(set(notes))

# Mapping
note_to_int = {note_value: number for number, note_value in enumerate(unique_notes)}
int_to_note = {number: note_value for number, note_value in enumerate(unique_notes)}

# Load trained model
model = load_model("music_model.h5")

sequence_length = 50

# Random starting point
start = np.random.randint(0, len(notes) - sequence_length)

pattern = notes[start:start + sequence_length]

prediction_output = []

# Generate 100 notes
for note_index in range(100):

    input_sequence = [note_to_int[n] for n in pattern]

    input_sequence = np.reshape(input_sequence, (1, sequence_length, 1))

    input_sequence = input_sequence / float(len(unique_notes))

    prediction = model.predict(input_sequence, verbose=0)

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern.append(result)

    pattern = pattern[1:]

# Create MIDI file
offset = 0
output_notes = []

for pattern_value in prediction_output:

    if '.' in pattern_value:
        notes_in_chord = pattern_value.split('.')
        notes_list = []

        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes_list.append(new_note)

        new_chord = chord.Chord(notes_list)
        new_chord.offset = offset
        output_notes.append(new_chord)

    else:
        new_note = note.Note(pattern_value)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)

midi_stream.write('midi', fp='generated_music.mid')

print("AI Music Generated Successfully!")