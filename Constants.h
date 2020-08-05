#pragma once

const char basedir[] = "./Database/BottomPhoto/photo"; //путь к фото
const char datadir[] = "./Database/BottomPhoto/data.txt"; //путь к данным по фото
const char tiledir[] = "./Database/BottomPhoto/Tiles"; //путь к базе тайлов
const char logdir[] = "./logs.csv"; //путь к логам

const int MAXPATHLENGTH = 64;
const int MaxScale = 19; //максимальное число уровней карты
const int patch_size = 256; //ширина тайла
const double max_half_width = 33554432; //половина максимальной ширины полотна карты, выраженна€ в пиксел€х

const float focus = 0.008; //фокусное рассто€ние камеры
const int px = 90; //количество пикселей на миллиметр матрицы камеры
const float Hnom = 1.6; //номинальна€ высота съемки 
const float mashtab = Hnom / focus; //масштаб снимка 1:m на номинальной высоте Hnom
const float PX = px * 1000 / mashtab; //количество пикселей на метр в масштабе изображени€, сделанного на высоте Hnom

const float gamma = 0.6; //коэффициент гамма-коррекции
const float claheLimit = 3.5; //граница контрастности дл€ автонастройки €ркости и контрастности снимков

enum SaveMode { SAVE, REPLACE };