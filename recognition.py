from tkinter import *
from PIL import Image, ImageDraw
from keras.models import load_model

import numpy as np


class HandwritingRecognition:
    def __init__(self):
        self.model = load_model('model.h5')

        self.root = Tk()
        self.root.geometry('600x650')
        self.root.title('Neural network')

        self.root.columnconfigure(3, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.canv = Canvas(self.root, bg='black')
        self.canv.grid(row=1, column=1, columnspan=7, padx=5, pady=5, sticky=E+W+S+N)

        ok_btn = Button(self.root, text='Ok', width=10, border=5, command=lambda: self.recognition())
        ok_btn.grid(row=2, column=1, padx=5, pady=5)

        reset_btn = Button(self.root, text='Reset', width=10, border=5, command=lambda: self.clear())
        reset_btn.grid(row=2, column=2, padx=5, pady=5)

        result_label = Label(self.root, text="Result: ")
        result_label.grid(row=3, column=1, padx=5, pady=5)

        self.result = Label(self.root)
        self.result.grid(row=3, column=2)

        self.brush_size = 25
        self.brush_color = 'white'

        self.image = Image.new('L', (600, 600), 'black')
        self.draw_instance = ImageDraw.Draw(self.image)

        self.canv.bind('<B1-Motion>', self.draw)

        self.root.mainloop()

    def draw(self, event):
        x1 = event.x - self.brush_size
        y1 = event.y - self.brush_size
        x2 = event.x + self.brush_size
        y2 = event.y + self.brush_size
        color = self.brush_color

        self.canv.create_oval(x1, y1, x2, y2, fill=color, outline=color)
        self.draw_instance.ellipse((x1, y1, x2, y2), fill=color, outline=color)

    def get_image(self):
        self.image.thumbnail((28, 28))
        matrix = np.array(self.image).astype('float32')
        matrix /= 255
        return matrix.reshape(1, 784)

    def clear(self):
        self.canv.delete('all')
        del self.draw_instance
        del self.image

        self.image = Image.new('L', (600, 600), 'black')
        self.draw_instance = ImageDraw.Draw(self.image)

    def recognition(self):
        test = self.model.predict_on_batch(self.get_image())
        self.result.config(text='{:d} - {:.2%}'.format(test.argmax(), test.max()))
        self.clear()


if __name__ == '__main__':
    HandwritingRecognition()
