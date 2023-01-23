from flask import Flask, url_for
from markupsafe import escape

app = Flask(__name__)


@app.route('/')
def index():
    return '<p>index</p>'


@app.route('/<name>')
def hello(name):
    return f'Hello, {escape(name)}!'


@app.route('/user/<username>')
def show_user_profile(username):
    return f'User {escape(username)}'


@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post {post_id}'


@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    return f'Subpath {escape(subpath)}'


@app.route('/projects/')
def projects():
    return '<p>The project page</p>'


@app.route('/about')
def about():
    return '<p>The about page</p>'


@app.route('/login')
def login():
    return '<p>login</p>'


@app.route('/user/<username>')
def profile(username):
    return f'<p>{username}\'s profile</p>'


with app.test_request_context():
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))
