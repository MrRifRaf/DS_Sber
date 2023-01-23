import os

from flask import Flask, flash, redirect, render_template, request, url_for

from inference import infer, load_models

INFER_FOLDER = os.path.join('static', 'infer')
t2, wg = load_models()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = INFER_FOLDER
SPEAKERS = {
    'Cori Samuel (female)': 0,
    'Phil Benson (male)': 1,
    'John Van Stan (male)': 2,
    'Mike Pelton (male)': 3,
    'Tony Oliva (male)': 4,
    'Maria Kasper (female)': 5,
    'Helen Taylor (female)': 6,
    'Sylviamb (female)': 7,
    'Celine Major (female)': 8,
    'LikeManyWaters (female)': 9
}


@app.route('/', methods=['GET', 'POST'])
def synthesize():
    if request.method == 'POST':
        error = None

        text = request.form.get('text_in')
        speaker = request.form.get('speaker')
        speaker_id = SPEAKERS.get(speaker, 0)
        if speaker_id < 0 or speaker_id > 9:
            error = 'Incorrect Speaker ID'

        if error is None:
            infer(t2, wg, text, speaker_id)
            return redirect(
                url_for('show_inference', text=text, speaker=speaker))

        flash(error)
    return render_template("index.html", speakers=SPEAKERS)


@app.route('/inference')
def show_inference():
    text = request.args['text']
    speaker = request.args['speaker']
    mel_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mel.png')
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.wav')
    return render_template('inference.html',
                           text=text,
                           speaker=speaker,
                           spectrogram=mel_path,
                           speech=audio_path)
