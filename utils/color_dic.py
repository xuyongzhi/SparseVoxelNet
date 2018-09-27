#https://jonasjacek.github.io/colors/
import numpy as np

color_2_rgb	= {
  'Red' 		  : (255,0,0),
  'Yellow' 		: (255,255,0),
  'Maroon' 		: (128,0,0),
  'Green' 		: (0,128,0),
  'Olive' 		: (128,128,0),
  'Navy' 		  : (0,0,128),
  'Purple' 		: (128,0,128),
  'Teal' 		  : (0,128,128),
  'Silver' 		: (192,192,192),
  'Grey' 		  : (128,128,128),
  'Lime' 		  : (0,255,0),
  'Blue' 		  : (0,0,255),
  'Fuchsia' 	: (255,0,255),
  'Aqua' 		  : (0,255,255),
  'White' 		: (255,255,255),
  'Black' 		: (0,0,0),

  'NavyBlue'          : (0,0,95),
  'DarkGreen'         : (0,95,0),
  'DarkRed'           : (95,0,0),
  'Purple4'           : (95,0,175),
  'BlueViolet'        : (95,0,255),
  'SpringGreen2'			: (0,215,135),
  'Chartreuse2'				: (95,255,0),
  'DarkMagenta'				: (135,0,175),
  'SteelBlue1'				: (95,215,255),
  'DarkViolet'				: (135,0,215),
  'Yellow4'				    : (135,135,0),
  'LightGreen'				: (135,255,95),
  'MediumVioletRed'		: (175,0,135),
  'DarkOrange3'				: (175,95,0),
  'LightSkyBlue1'			: (175,215,255),
  'DarkGoldenrod'			: (175,135,0),
  'Magenta2'				  : (215,0,255),
  'MediumPurple1'			: (175,135,255),
  'Plum2 				'			: (215,175,255),
  'GreenYellow'				: (175,255,0),
  'GreenYellow'				: (175,255,0),
  'Orange1'				    : (255,175,0),
  'Salmon1'           : (255,135,95),
  'NavajoWhite1'			: (255,215,175),
  'Wheat1'				    : (255,255,175),
  'Grey30'				    : (78,78,78),
  'Grey78'				    : (198,198,198),
}

color_order_0 = ['Red','Green','Yellow','Blue','Red','Olive','Navy','Maroon',\
                 'Purple','Teal','Silver','Grey','Lime','Fuchsia','Aqua','Black','White']

color_order = color_order_0 + [c for c in color_2_rgb.keys() if c not in color_order_0]
rgb_order = np.array([color_2_rgb[c] for c in color_order], dtype=np.uint8)


