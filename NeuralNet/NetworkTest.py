import pygame
import mnist_loader
import network
import button
import numpy as np
from PIL import Image

def main():
	pygame.init()

	win = pygame.display.set_mode((500, 500))
	pygame.display.set_caption("hopefully this works")
	clock = pygame.time.Clock()
	#train_button = pygame.Rect(100, 50, 50, 50)	
	train_button = button.Button( (255, 0, 0), 110, 50, 50, 50, "TRAIN" )
	guess_button = button.Button( (255, 0, 0), 340, 50, 50, 50, "GUESS")
	class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	#default button title will be "TRAIN"
	#drawing_screen = pygame.Rect(110, 110, 280, 280)
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	training_data = list(training_data)
	test_data = list(test_data)

	net = network.Network([784, 30, 10])

	run = True  

	while run:
		win.fill((255,255,255))
		train_button.draw(win, (0, 0, 0))
		guess_button.draw(win, (0, 0, 0))
		pygame.draw.rect(win, [0, 0, 0], [108, 108, 284, 284], 2) #draws the drawing screen for user input
		#pygame.draw.rect(win, [255, 255, 255], drawing_screen)
		pygame.display.update()

		for event in pygame.event.get():
			mouse_pos = pygame.mouse.get_pos()

			if event.type == pygame.QUIT:
				run = False

			if event.type == pygame.MOUSEBUTTONDOWN:
				if train_button.isOver(mouse_pos) and train_button.text != "TRAINED":
					#change the button title to "TRAINING"
					#train_button.text = "TRAINING"
					net.SGD(training_data, 2, 10, 3.0, test_data = test_data)
					#change the button title to "TRAINED"
					train_button.text = "TRAINED"
					#print("Training complete!")
				elif guess_button.isOver(mouse_pos): #need to figure out how to apply np.reshape so i can feed the user_inputs in...
					im = Image.open('my_test.png', 'r')
					#https://www.dreamincode.net/forums/topic/269925-create-image-file-from-pixel-value-python/ 
					#use ^ to create userinput from the pygame drawing screen...
					pixel_values = list(im.getdata())
					pixel_values_flat = [x for sets in pixel_values for x in sets]

					inputs = list()
					for i in range(784):
						grayscale_val = pixel_values_flat[i*3]
						inputs.append((255 - grayscale_val) / 255.0)

					user_inputs = [np.reshape(inputs, (784, 1))]

					#print(new_pixel_values_flat)
					print("Prediction: " + class_names[np.argmax(net.feedforward(user_inputs[0]))])
					#print("Prediction: " + class_names[np.argmax(net.feedforward(training_data[40]))])

			if event.type == pygame.MOUSEMOTION:
				if train_button.isOver(mouse_pos):
					train_button.color = (200, 0, 0)
				elif guess_button.isOver(mouse_pos):
					guess_button.color = (200, 0, 0)
				else:
					train_button.color = (255, 0, 0)
					guess_button.color = (255, 0, 0)


		#win.fill((0,0,0))
		#pygame.draw.rect(win, [255, 0, 0], train_button)
		#pygame.draw.rect(win, [255, 255, 255], drawing_screen)

		pygame.display.update()
		clock.tick(60)

if __name__ == '__main__':
	main()
	pygame.quit()