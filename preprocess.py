import os
from music21 import converter, note, chord

midi_folder = "midi_songs"
files = os.listdir(midi_folder)

print("Files found:", files)

notes = []

for file in files:
    if file.endswith(".mid"):
        path = os.path.join(midi_folder, file)
        print("Processing:", path)

        midi = converter.parse(path)

        elements = midi.flat.notes

        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.pitches))

        print("Success:", file)

print("Total notes:", len(notes))
print(notes[:50])

import pickle

with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)