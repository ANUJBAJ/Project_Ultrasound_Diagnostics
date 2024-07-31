from flask import Flask,render_template,send_file,request,redirect,url_for
import os
app = Flask(__name__)
Upload_folder = 'uploads'
if not os.path.exists(Upload_folder):
    os.makedirs(Upload_folder)
app.config['Upload_folder'] = Upload_folder
@app.route('/')
def base():
    return render_template('frontend.html')
@app.route('/upload',methods = ['POST'])
def upload_file():
    
    if 'file' not in request.files:
        print('helloiamhere')
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    if file :
        if file.filename.split('.')[-1] == 'csv':
            file_path = os.path.join(app.config['Upload_folder'],file.filename)
            file.save(file_path)
        else:
            return render_template('error.html')
       
        return redirect(url_for('download_file',filename = file.filename))
@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['Upload_folder'] , filename)
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)