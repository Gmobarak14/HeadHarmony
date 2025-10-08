{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 9,
			"minor" : 0,
			"revision" : 7,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 423.0, 715.0, 939.0, 912.0 ],
		"gridsize" : [ 15.0, 15.0 ],
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-63",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 217.0, 145.0, 21.0, 22.0 ],
					"text" : "py"
				}

			}
, 			{
				"box" : 				{
					"focusbordercolor" : [ 0.57636836783545, 0.576368229540612, 0.576368265679262, 1.0 ],
					"fontface" : 1,
					"fontsize" : 16.0,
					"id" : "obj-36",
					"maxclass" : "live.slider",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 563.0, 1045.0, 98.0, 122.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 336.0, 574.0, 208.0, 129.0 ],
					"saved_attribute_attributes" : 					{
						"focusbordercolor" : 						{
							"expression" : "themecolor.live_value_arc_zombie"
						}
,
						"valueof" : 						{
							"parameter_longname" : "Transpose",
							"parameter_mmax" : 12.0,
							"parameter_mmin" : -12.0,
							"parameter_modmode" : 4,
							"parameter_shortname" : "Trasnpose",
							"parameter_type" : 1,
							"parameter_unitstyle" : 0
						}

					}
,
					"varname" : "live.slider"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 18.0,
					"id" : "obj-19",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 469.5, 574.0, 220.0, 27.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 242.0, 420.0, 191.0, 27.0 ],
					"style" : "velvet",
					"text" : "HEAD POSITION"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-78",
					"maxclass" : "gswitch2",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 791.549306154251099, 821.126771330833435, 54.929578185081482, 46.4788738489151 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-77",
					"maxclass" : "gswitch2",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 654.929586052894592, 821.126771330833435, 54.929578185081482, 46.4788738489151 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-76",
					"maxclass" : "gswitch2",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 477.464794993400574, 825.352123498916626, 54.929578185081482, 46.4788738489151 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-75",
					"maxclass" : "gswitch2",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 208.450706958770752, 838.028180003166199, 54.929578185081482, 46.4788738489151 ]
				}

			}
, 			{
				"box" : 				{
					"coll_data" : 					{
						"count" : 7,
						"data" : [ 							{
								"key" : 1,
								"value" : [ 60 ]
							}
, 							{
								"key" : 2,
								"value" : [ 62 ]
							}
, 							{
								"key" : 3,
								"value" : [ 64 ]
							}
, 							{
								"key" : 4,
								"value" : [ 65 ]
							}
, 							{
								"key" : 5,
								"value" : [ 67 ]
							}
, 							{
								"key" : 6,
								"value" : [ 69 ]
							}
, 							{
								"key" : 7,
								"value" : [ 71 ]
							}
 ]
					}
,
					"id" : "obj-72",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "", "", "", "" ],
					"patching_rect" : [ 711.25, 939.436632037162781, 59.0, 22.0 ],
					"saved_object_attributes" : 					{
						"embed" : 1,
						"precision" : 50
					}
,
					"text" : "coll notes"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 20.0,
					"id" : "obj-71",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 129.0, 629.0, 218.0, 29.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 517.0, 473.0, 88.0, 29.0 ],
					"text" : "Down"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 20.0,
					"id" : "obj-70",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 114.0, 614.0, 218.0, 29.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 357.0, 473.0, 88.0, 29.0 ],
					"text" : "Up"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 20.0,
					"id" : "obj-69",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 99.0, 599.0, 218.0, 29.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 197.0, 473.0, 88.0, 29.0 ],
					"text" : "Right"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 20.0,
					"id" : "obj-68",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 84.0, 584.0, 218.0, 29.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 43.0, 473.0, 88.0, 29.0 ],
					"text" : "Left"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-62",
					"items" : [ "None", ",", "I", "maj", ",", "ii", "min", ",", "iii", "min", ",", "IV", "maj", ",", "V", "maj", ",", "vi", "min", ",", "vii", "dim" ],
					"maxclass" : "umenu",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 804.0, 778.0, 100.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 517.0, 523.0, 96.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-60",
					"items" : [ "None", ",", "I", "maj", ",", "ii", "min", ",", "iii", "min", ",", "IV", "maj", ",", "V", "maj", ",", "vi", "min", ",", "vii", "dim" ],
					"maxclass" : "umenu",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 648.0, 778.125, 100.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 353.0, 523.0, 96.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-59",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 200.0, 1057.8125, 29.5, 22.0 ],
					"text" : "+ 0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-40",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 376.5625, 815.625, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-58",
					"items" : [ "None", ",", "I", "maj", ",", "ii", "min", ",", "iii", "min", ",", "IV", "maj", ",", "V", "maj", ",", "vi", "min", ",", "vii", "dim" ],
					"maxclass" : "umenu",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 496.0, 778.125, 100.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 197.0, 523.0, 96.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-39",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 112.5, 867.1875, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-32",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 406.25, 976.5625, 32.0, 22.0 ],
					"text" : "print"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-24",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 0,
					"patching_rect" : [ 200.0, 1171.875, 49.0, 22.0 ],
					"text" : "noteout"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-26",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "float", "float" ],
					"patching_rect" : [ 200.0, 1098.4375, 108.0, 22.0 ],
					"text" : "makenote 120 600"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-22",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 204.6875, 1007.8125, 35.0, 22.0 ],
					"text" : "iter 3"
				}

			}
, 			{
				"box" : 				{
					"coll_data" : 					{
						"count" : 7,
						"data" : [ 							{
								"key" : 1,
								"value" : [ 60, 64, 67 ]
							}
, 							{
								"key" : 2,
								"value" : [ 62, 65, 69 ]
							}
, 							{
								"key" : 3,
								"value" : [ 64, 67, 71 ]
							}
, 							{
								"key" : 4,
								"value" : [ 65, 69, 72 ]
							}
, 							{
								"key" : 5,
								"value" : [ 67, 71, 74 ]
							}
, 							{
								"key" : 6,
								"value" : [ 69, 72, 76 ]
							}
, 							{
								"key" : 7,
								"value" : [ 71, 74, 77 ]
							}
 ]
					}
,
					"id" : "obj-16",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "", "", "", "" ],
					"patching_rect" : [ 204.6875, 951.5625, 66.0, 22.0 ],
					"saved_object_attributes" : 					{
						"embed" : 1,
						"precision" : 40
					}
,
					"text" : "coll chords"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-9",
					"items" : [ "None", ",", "I", "maj", ",", "ii", "min", ",", "iii", "min", ",", "IV", "maj", ",", "V", "maj", ",", "vi", "min", ",", "vii", "dim" ],
					"maxclass" : "umenu",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 200.0, 787.0, 100.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 43.0, 523.0, 100.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-61",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1502.325527667999268, 218.60464334487915, 150.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 157.0, 629.0, 150.0, 20.0 ],
					"text" : "Root Note Only"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-49",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 2178.048832416534424, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 761.0, 294.0, 68.0, 59.0 ],
					"text" : "Ti",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-50",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 2081.39527440071106, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 643.0, 294.0, 68.0, 59.0 ],
					"text" : "La",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-51",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1969.767371416091919, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 529.0, 294.0, 68.0, 59.0 ],
					"text" : "So",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-52",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1851.162724494934082, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 411.0, 294.0, 68.0, 59.0 ],
					"text" : "Fa",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-53",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1732.558077573776245, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 294.0, 294.0, 68.0, 59.0 ],
					"text" : "Mi",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-54",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1609.302268028259277, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 171.0, 294.0, 68.0, 59.0 ],
					"text" : "Re",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-55",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1490.69762110710144, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 51.0, 294.0, 68.0, 59.0 ],
					"text" : "Do",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-48",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 32.39436662197113, 777.125, 24.0, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 74.0, 610.0, 57.0, 57.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-44",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 2013.953416347503662, 227.906968593597412, 29.5, 22.0 ],
					"text" : "71"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-43",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1967.441790103912354, 227.906968593597412, 29.5, 22.0 ],
					"text" : "69"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-42",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1925.581326484680176, 227.906968593597412, 29.5, 22.0 ],
					"text" : "67"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-41",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1883.720862865447998, 227.906968593597412, 29.5, 22.0 ],
					"text" : "65"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-38",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1839.534817934036255, 227.906968593597412, 29.5, 22.0 ],
					"text" : "64"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-34",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1795.348773002624512, 227.906968593597412, 29.5, 22.0 ],
					"text" : "62"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-27",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1751.162728071212769, 227.906968593597412, 29.5, 22.0 ],
					"text" : "60"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-7",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 726.0, 544.0, 29.5, 22.0 ],
					"text" : "+ 0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-12",
					"maxclass" : "newobj",
					"numinlets" : 5,
					"numoutlets" : 5,
					"outlettype" : [ "", "", "", "", "" ],
					"patching_rect" : [ 255.813944339752197, 486.046494245529175, 264.0, 22.0 ],
					"text" : "route /head/left /head/right /head/up /head/down"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-11",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"patching_rect" : [ 265.116269588470459, 351.16277813911438, 63.0, 22.0 ],
					"text" : "unpack s f"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-6",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 248.837200403213501, 297.674407958984375, 97.0, 22.0 ],
					"text" : "udpreceive 7400"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-33",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1372.092974185943604, 211.627899408340454, 55.0, 22.0 ],
					"text" : "71 74 77"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-35",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1360.465067625045776, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 761.0, 208.0, 68.0, 59.0 ],
					"text" : "vii dim",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-28",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1255.813908576965332, 211.627899408340454, 55.0, 22.0 ],
					"text" : "69 72 76"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-30",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1241.860420703887939, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 643.0, 208.0, 68.0, 59.0 ],
					"text" : "vi min",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-23",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1141.860424280166626, 211.627899408340454, 55.0, 22.0 ],
					"text" : "67 71 74"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-25",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1127.906936407089233, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 529.0, 208.0, 68.0, 59.0 ],
					"text" : "V maj",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-18",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1023.255777359008789, 211.627899408340454, 55.0, 22.0 ],
					"text" : "65 69 72"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-20",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1009.302289485931396, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 411.0, 208.0, 68.0, 59.0 ],
					"text" : "IV maj",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-13",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 906.976711750030518, 211.627899408340454, 55.0, 22.0 ],
					"text" : "64 67 71"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-15",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 893.023223876953125, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 294.0, 208.0, 68.0, 59.0 ],
					"text" : "iii min",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-8",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 783.72090220451355, 211.627899408340454, 55.0, 22.0 ],
					"text" : "62 65 69"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-10",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 769.767414331436157, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 171.0, 208.0, 68.0, 59.0 ],
					"text" : "ii min",
					"texton" : "Cmaj on"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 672.790673971176147, 469.767425060272217, 35.0, 22.0 ],
					"text" : "iter 3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 0,
					"patching_rect" : [ 738.0, 666.0, 49.0, 22.0 ],
					"text" : "noteout"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 662.790673971176147, 211.627899408340454, 55.0, 22.0 ],
					"text" : "60 64 67"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "float", "float" ],
					"patching_rect" : [ 722.0, 612.0, 108.0, 22.0 ],
					"text" : "makenote 120 600"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-2",
					"maxclass" : "textbutton",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 651.16276741027832, 113.953484296798706, 68.0, 59.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 51.0, 208.0, 68.0, 59.0 ],
					"text" : "I maj",
					"texton" : "Cmaj on"
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 1 ],
					"source" : [ "obj-1", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 0 ],
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 0 ],
					"source" : [ "obj-10", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"source" : [ "obj-11", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-58", 0 ],
					"source" : [ "obj-12", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-60", 0 ],
					"source" : [ "obj-12", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-62", 0 ],
					"source" : [ "obj-12", 3 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-9", 0 ],
					"source" : [ "obj-12", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-13", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-13", 0 ],
					"source" : [ "obj-15", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-22", 0 ],
					"order" : 1,
					"source" : [ "obj-16", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-32", 0 ],
					"order" : 0,
					"source" : [ "obj-16", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-18", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4", 0 ],
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-18", 0 ],
					"source" : [ "obj-20", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-59", 0 ],
					"source" : [ "obj-22", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-23", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-23", 0 ],
					"source" : [ "obj-25", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-24", 1 ],
					"source" : [ "obj-26", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-24", 0 ],
					"source" : [ "obj-26", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-27", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-28", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-3", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-28", 0 ],
					"source" : [ "obj-30", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-33", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-34", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-33", 0 ],
					"source" : [ "obj-35", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-59", 1 ],
					"order" : 1,
					"source" : [ "obj-36", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 1 ],
					"order" : 0,
					"source" : [ "obj-36", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-38", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-4", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-41", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-42", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-43", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-44", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-75", 0 ],
					"order" : 3,
					"source" : [ "obj-48", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-76", 0 ],
					"order" : 2,
					"source" : [ "obj-48", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 0 ],
					"order" : 1,
					"source" : [ "obj-48", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-78", 0 ],
					"order" : 0,
					"source" : [ "obj-48", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-44", 0 ],
					"source" : [ "obj-49", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 0 ],
					"source" : [ "obj-50", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-42", 0 ],
					"source" : [ "obj-51", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-41", 0 ],
					"source" : [ "obj-52", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-38", 0 ],
					"source" : [ "obj-53", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-34", 0 ],
					"source" : [ "obj-54", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-27", 0 ],
					"source" : [ "obj-55", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-40", 0 ],
					"order" : 1,
					"source" : [ "obj-58", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-76", 1 ],
					"order" : 0,
					"source" : [ "obj-58", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-26", 0 ],
					"source" : [ "obj-59", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-11", 0 ],
					"source" : [ "obj-6", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 1 ],
					"source" : [ "obj-60", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-78", 1 ],
					"source" : [ "obj-62", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-7", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-59", 0 ],
					"source" : [ "obj-72", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 0 ],
					"source" : [ "obj-75", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-72", 0 ],
					"source" : [ "obj-75", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 0 ],
					"source" : [ "obj-76", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-72", 0 ],
					"source" : [ "obj-76", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 0 ],
					"source" : [ "obj-77", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-72", 0 ],
					"source" : [ "obj-77", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 0 ],
					"source" : [ "obj-78", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-72", 0 ],
					"source" : [ "obj-78", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-8", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-39", 0 ],
					"order" : 1,
					"source" : [ "obj-9", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-75", 1 ],
					"order" : 0,
					"source" : [ "obj-9", 0 ]
				}

			}
 ],
		"parameters" : 		{
			"obj-36" : [ "Transpose", "Trasnpose", 0 ],
			"parameterbanks" : 			{
				"0" : 				{
					"index" : 0,
					"name" : "",
					"parameters" : [ "-", "-", "-", "-", "-", "-", "-", "-" ]
				}

			}
,
			"inherited_shortname" : 1
		}
,
		"dependency_cache" : [  ],
		"autosave" : 0
	}

}
