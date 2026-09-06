import os
import uuid
from pathlib import Path

from flask import Flask, render_template, send_from_directory
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from PIL import Image
from werkzeug.utils import secure_filename
from wtforms import FileField, FloatField, SubmitField

from stylize import load_models, transfer


ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'change-this-for-production')
app.config['UPLOAD_FOLDER'] = ROOT / 'static' / 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True, parents=True)


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


encoder, decoder, device = load_models(ROOT / 'models' / 'vgg_normalised.pth', ROOT / 'artifacts' / 'decoder.pth')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_upload(file):
    filename = secure_filename(file.filename)
    if not filename or not allowed_file(filename):
        raise ValueError('Images must be PNG, JPG, or JPEG files.')
    filename = f'{uuid.uuid4().hex}_{filename}'
    file.save(app.config['UPLOAD_FOLDER'] / filename)
    return filename


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = content_filename = style_filename = error = None
    if form.validate_on_submit():
        try:
            if not form.content.data or not form.content.data.filename or not form.style.data or not form.style.data.filename:
                raise ValueError('Please upload both a content image and a style image.')
            if form.alpha.data is None or not 0 <= form.alpha.data <= 1:
                raise ValueError('Style strength must be between 0 and 1.')
            content_filename = save_upload(form.content.data)
            style_filename = save_upload(form.style.data)
            output = transfer(
                Image.open(app.config['UPLOAD_FOLDER'] / content_filename),
                Image.open(app.config['UPLOAD_FOLDER'] / style_filename),
                encoder, decoder, device, form.alpha.data,
            )
            result_image = f'stylized_{uuid.uuid4().hex}.jpg'
            output.save(app.config['UPLOAD_FOLDER'] / result_image, quality=95)
        except Exception as exc:
            error = str(exc)
    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(ROOT / 'examples', filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
