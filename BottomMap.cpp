#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <ctime>
#include "Constants.h"
#include <math.h>
#include <map>

using namespace cv;
using namespace std;

double getValue(string src, const char* val, int size, char delimetr)
{
	double X = 0;
	string subbuff = "";
	int n1 = 0;
	int n2 = 0;

	n1 = src.find(val);
	n2 = src.find_first_of(delimetr, n1);
	if (n1 != string::npos && n2 != string::npos)
		subbuff = src.substr(n1 + size, n2 - (n1 + size));
	X = stod(subbuff);

	return X;
}
Mat gammaCorrection(Mat img, const double gamma)
{
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	Mat res = img.clone();
	LUT(img, lookUpTable, res);

	return res;
}

class SBuffer
{
private:
	map <pair <int, int>, Mat> tileStack;
	map <pair <int, int>, Mat>::iterator it;
	int scale;
public:
	SBuffer(int scale);
	~SBuffer();

	int addTile(Mat tile, Point pnt, SaveMode mode);
	Mat getTile(Point pnt);
	int write(int min_x, int max_x, int min_y, int max_y);
	int writeAll();
};

class Tiles
{
private:
	Size s;
	Mat* tile;
	int scale;
	int rows;
	int cols;
	Point ULpnt;

	int crop(Mat A);
	Mat normalize(Mat A, double X, double Y, double H, double angle);
	Mat rescale(Mat A, float scale_factor);
public:
	int save();
	int replace();
	int saveToBuff(SBuffer* buff, SaveMode mode);
	Tiles(Mat A, int scale, double X, double Y, double H, double angle);
	Tiles(int min_x, int min_y, int rows, int cols, int base_scale);
	Tiles(int min_x, int min_y, int rows, int cols, int base_scale, SBuffer* buff);
	~Tiles();

	Point giveULpnt();
	int giveScale();
	Mat giveTile(int x, int y);
	int giveRows();
	int giveCols();
};

int main()
{
	int all_tiles = 0;
	float all_time = 0;

	ifstream file;
	file.open(datadir, ifstream::in);

	ofstream out;
	remove(logdir);
	out.open(logdir);
	out << "name; startscale (sec); downscale (sec); all process (sec); written;" << endl;

	string logs = "";
	string name;
	double X, Y, H, angle;

	string buff;
	int min_x;
	int min_y;
	int max_x;
	int max_y;

	SBuffer* buffer[MaxScale]; //Создание динамического буфера для сохранения тайлов
	for (int i = 0; i < MaxScale; i++)
	{
		buffer[i] = new SBuffer(i);
	}

	while (getline(file, buff))
	{
		X = getValue(buff, "X(m): ", 6, ',');
		Y = getValue(buff, "Y(m): ", 6, ',');
		H = getValue(buff, "Alt(m): ", 8, ',');
		angle = getValue(buff, "Heading(deg): ", 14, ',');
		name = buff.substr(0, buff.find_first_of(' '));

		logs += name + ";";

		char path[MAXPATHLENGTH];
		memset(path, 0, 64);
		sprintf_s(path, 64, "%s/%s", basedir, name.c_str());
		Mat base = imread(path, IMREAD_UNCHANGED);

		Tiles* t[MaxScale];
		int start_s = 17;
		int last_s = 0;
		int tile_count = 0;

		//Cоздание тайлов начального масштаба
		clock_t st = clock();
		clock_t mid = st;
		clock_t end;

		t[start_s] = new Tiles(base, start_s, X, Y, H, angle);
		min_x = t[start_s]->giveULpnt().x;
		max_x = t[start_s]->giveULpnt().x + t[start_s]->giveRows();
		min_y = t[start_s]->giveULpnt().y;
		max_y = t[start_s]->giveULpnt().y + t[start_s]->giveCols();

		tile_count += buffer[start_s]->write(min_x, max_x, min_y, max_y);
		t[start_s]->saveToBuff(buffer[start_s], SAVE);

		end = clock();
		logs += to_string((float)(end - mid) / CLOCKS_PER_SEC) + ";";

		//Создание тайлов меньшего масштаба
		for (int k = start_s - 1; k >= last_s; k--)
		{
			t[k] = new Tiles(t[k + 1]->giveULpnt().x, t[k + 1]->giveULpnt().y,
				t[k + 1]->giveRows(), t[k + 1]->giveCols(), k + 1, buffer[k + 1]);

			min_x = t[k]->giveULpnt().x;
			max_x = t[k]->giveULpnt().x + t[k]->giveRows();
			min_y = t[k]->giveULpnt().y;
			max_y = t[k]->giveULpnt().y + t[k]->giveCols();

			tile_count += buffer[k]->write(min_x, max_x, min_y, max_y);
			t[k]->saveToBuff(buffer[k], REPLACE);
			delete t[k + 1];
		}
		delete t[last_s];
		mid = end;
		end = clock();
		logs += to_string((float)(end - mid) / CLOCKS_PER_SEC) + ";";

		end = clock();
		logs += to_string((float)(end - st) / CLOCKS_PER_SEC) + ";" + to_string(tile_count) + ";";
		out << logs << endl;
		logs = "";

		all_time += (float)(end - st) / CLOCKS_PER_SEC;
		all_tiles += tile_count;

		//Если нужно обрабатывать каждый второй снимок
		getline(file, buff);
	}

	cout << all_time << endl << all_tiles << endl;

	for (int i = 0; i < MaxScale; i++)
	{
		buffer[i]->writeAll();
	}

	for (int i = 0; i < MaxScale; i++)
	{
		delete buffer[i];
	}

	out.close();
	return 0;
}

//Определения класса Tiles

Tiles::Tiles(Mat A, int scale, double X, double Y, double H, double angle)
{
	Mat B = normalize(A, X, Y, H, angle);

	s = Size(patch_size, patch_size);
	this->scale = scale;
	rows = B.rows / patch_size + (B.rows % patch_size > 0);
	cols = B.cols / patch_size + (B.cols % patch_size > 0);
	tile = new Mat[(rows * cols)];
	int n = crop(B);
}
Tiles::Tiles(int min_x, int min_y, int height, int width, int base_scale)
{
	s = Size(patch_size, patch_size);

	ULpnt = Point(min_x, min_y) / 2;
	Point Endpnt = Point(min_x + width, min_y + height) / 2;
	rows = Endpnt.y - ULpnt.y + 1;
	cols = Endpnt.x - ULpnt.x + 1;
	scale = base_scale - 1;
	tile = new Mat[rows * cols];
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int x = (j + ULpnt.x) * 2;
			int y = (i + ULpnt.y) * 2;
			char path[MAXPATHLENGTH];

			memset(path, 0, 64);
			sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, base_scale, x, y);
			Mat ULimg = imread(path, IMREAD_UNCHANGED);

			memset(path, 0, 64);
			sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, base_scale, x + 1, y);
			Mat URimg = imread(path, IMREAD_UNCHANGED);

			memset(path, 0, 64);
			sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, base_scale, x, y + 1);
			Mat DLimg = imread(path, IMREAD_UNCHANGED);

			memset(path, 0, 64);
			sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, base_scale, x + 1, y + 1);
			Mat DRimg = imread(path, IMREAD_UNCHANGED);

			Rect rect1 = Rect(0, 0, patch_size, patch_size);
			Rect rect2 = Rect(patch_size, 0, patch_size, patch_size);
			Rect rect3 = Rect(0, patch_size, patch_size, patch_size);
			Rect rect4 = Rect(patch_size, patch_size, patch_size, patch_size);

			Mat buffim = Mat::zeros(patch_size * 2, patch_size * 2, 0);

			Mat ULroi = Mat(buffim, rect1);
			Mat URroi = Mat(buffim, rect2);
			Mat DLroi = Mat(buffim, rect3);
			Mat DRroi = Mat(buffim, rect4);

			if (!ULimg.empty()) ULimg.copyTo(ULroi);
			if (!URimg.empty()) URimg.copyTo(URroi);
			if (!DLimg.empty()) DLimg.copyTo(DLroi);
			if (!DRimg.empty()) DRimg.copyTo(DRroi);

			tile[i * cols + j] = rescale(buffim, 0.5);
		}
	}
}
Tiles::Tiles(int min_x, int min_y, int height, int width, int base_scale, SBuffer* buff)
{
	s = Size(patch_size, patch_size);

	ULpnt = Point(min_x, min_y) / 2;
	Point Endpnt = Point(min_x + width + 1, min_y + height + 1) / 2;
	rows = Endpnt.y - ULpnt.y;
	cols = Endpnt.x - ULpnt.x;
	scale = base_scale - 1;
	tile = new Mat[rows * cols];
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int x = (j + ULpnt.x) * 2;
			int y = (i + ULpnt.y) * 2;
			char path[MAXPATHLENGTH];

			memset(path, 0, 64);
			sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, base_scale, x, y);
			Mat ULimg = buff->getTile(Point(x, y));
			if (ULimg.empty()) ULimg = imread(path, IMREAD_UNCHANGED);

			memset(path, 0, 64);
			sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, base_scale, x + 1, y);
			Mat URimg = buff->getTile(Point(x + 1, y));
			if (URimg.empty()) URimg = imread(path, IMREAD_UNCHANGED);

			memset(path, 0, 64);
			sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, base_scale, x, y + 1);
			Mat DLimg = buff->getTile(Point(x, y + 1));
			if (DLimg.empty()) DLimg = imread(path, IMREAD_UNCHANGED);

			memset(path, 0, 64);
			sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, base_scale, x + 1, y + 1);
			Mat DRimg = buff->getTile(Point(x + 1, y + 1));
			if (DRimg.empty()) DRimg = imread(path, IMREAD_UNCHANGED);

			Rect rect1 = Rect(0, 0, patch_size, patch_size);
			Rect rect2 = Rect(patch_size, 0, patch_size, patch_size);
			Rect rect3 = Rect(0, patch_size, patch_size, patch_size);
			Rect rect4 = Rect(patch_size, patch_size, patch_size, patch_size);

			Mat buffim = Mat::zeros(patch_size * 2, patch_size * 2, 0);

			Mat ULroi = Mat(buffim, rect1);
			Mat URroi = Mat(buffim, rect2);
			Mat DLroi = Mat(buffim, rect3);
			Mat DRroi = Mat(buffim, rect4);

			if (!ULimg.empty()) ULimg.copyTo(ULroi);
			if (!URimg.empty()) URimg.copyTo(URroi);
			if (!DLimg.empty()) DLimg.copyTo(DLroi);
			if (!DRimg.empty()) DRimg.copyTo(DRroi);

			tile[i * cols + j] = rescale(buffim, 0.5);
		}
	}
}
Tiles::~Tiles()
{
	for (int i = 0; i < rows * cols; i++)
	{
		tile[i].release();
	}
}

int Tiles::save()
{
	int count = 0;
	Mat buffim;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			char path[MAXPATHLENGTH];
			memset(path, 0, 64);
			int l = sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, scale, j + ULpnt.x, i + ULpnt.y);
			buffim = imread(path, IMREAD_UNCHANGED);
			if (buffim.empty())
			{
				if (imwrite(path, tile[i * cols + j])) count++;
			}
			else
			{
				Mat gs_tile = Mat::zeros(s, 0);
				Mat mask = Mat::zeros(s, 0);
				Mat mask_inv = Mat::zeros(s, 0);
				Mat tile1 = Mat::zeros(s, buffim.type());
				Mat tile2 = Mat::zeros(s, buffim.type());
				Mat dst_tile = Mat::zeros(s, buffim.type());

				if (tile[i * cols + j].channels() == 3)
					cvtColor(tile[i * cols + j], gs_tile, COLOR_BGR2GRAY);
				else
					gs_tile = tile[i * cols + j];

				threshold(gs_tile, mask, 10, 255, THRESH_BINARY);
				bitwise_not(mask, mask_inv);

				bitwise_and(buffim, buffim, tile1, mask_inv);
				bitwise_and(tile[i * cols + j], tile[i * cols + j], tile2, mask);

				add(tile1, tile2, dst_tile);

				if (imwrite(path, dst_tile)) count++;
			}

		}
	}

	return count;
}
int Tiles::replace()
{
	int count = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			char path[MAXPATHLENGTH];
			memset(path, 0, 64);
			int l = sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, scale, j + ULpnt.x, i + ULpnt.y);
			if (imwrite(path, tile[i * cols + j])) count++;
		}
	}

	return count;
}
int Tiles::saveToBuff(SBuffer* buff, SaveMode mode)
{
	int count = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			Point pnt = Point(j + ULpnt.x, i + ULpnt.y);
			count += buff->addTile(tile[i * cols + j], pnt, mode);
		}
	}

	return count;
}
int Tiles::crop(Mat A)
{
	Mat plane = Mat::zeros(Size(cols * patch_size, rows * patch_size), A.type());
	Mat roi = Mat(plane, Rect(0, 0, A.cols, A.rows));
	A.copyTo(roi);
	int count = 0;
	Point2f corner;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			corner = Point2f(patch_size * j, patch_size * i);
			roi = Mat(plane, Rect(corner, s));
			roi.copyTo(tile[i * cols + j]);
			count++;
		}
	}
	return count;
}
Mat Tiles::rescale(Mat A, float scale_factor)
{
	Size s = Size(A.size().width * scale_factor, A.size().height * scale_factor);
	Mat B = Mat::zeros(s, A.type());
	resize(A, B, s);

	return B;
}
Mat Tiles::normalize(Mat A, double X, double Y, double H, double angle)
{
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(claheLimit);

	Mat A2;
	clahe->apply(A, A2);

	A2 = gammaCorrection(A2, gamma);

	//Масштабируем изображение в зависимости от высоты H
	Mat A1 = rescale(A2, Hnom / H);

	//Определяем размер контейнера B нового изображения с учетом поворота
	int new_size = 2 * (int)sqrt(pow(A1.rows / 2, 2) + pow(A1.cols / 2, 2));

	//Выражаем координаты изображения в пикселях и переносим к центру тайловой карты
	//чтобы избежать отрицательных чисел
	int px_x = (int)(max_half_width + X * PX);
	int px_y = (int)(max_half_width - Y * PX);

	//Размер контейнера B дополнительно расширяется, чтобы сопоставить 
	//верхний левый угол изображения с углом ближайшего тайла
	int d_cols = (px_x - new_size / 2) % patch_size;
	int d_rows = (px_y - new_size / 2) % patch_size;

	//Записываем координаты верхнего левого тайла
	ULpnt = Point((px_x - new_size / 2) / patch_size,
		(px_y - new_size / 2) / patch_size);

	//Создаем пустой контейнер B нового изображения
	Mat B = Mat::zeros(Size(new_size + d_cols, new_size + d_rows), A1.type());

	int x_trans = d_cols + (new_size - A1.cols) / 2;
	int y_trans = d_rows + (new_size - A1.rows) / 2;

	//Получаем одиночные матрицы двух последовательных линейных преобразований:
	//сдвиг и поворот
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, x_trans, 0, 1, y_trans);
	Mat rotate_mat = getRotationMatrix2D(Point(d_cols + new_size / 2, d_rows + new_size / 2), -angle, 1);


	//Расширяем матрицы до размера 3*3
	Mat trans_mat_3 = Mat::eye(Size(3, 3), trans_mat.type());
	Mat rotate_mat_3 = Mat::eye(Size(3, 3), rotate_mat.type());

	trans_mat_3.at<double>(0, 0) = trans_mat.at<double>(0, 0);
	trans_mat_3.at<double>(1, 0) = trans_mat.at<double>(1, 0);
	trans_mat_3.at<double>(0, 1) = trans_mat.at<double>(0, 1);
	trans_mat_3.at<double>(1, 1) = trans_mat.at<double>(1, 1);
	trans_mat_3.at<double>(0, 2) = trans_mat.at<double>(0, 2);
	trans_mat_3.at<double>(1, 2) = trans_mat.at<double>(1, 2);

	rotate_mat_3.at<double>(0, 0) = rotate_mat.at<double>(0, 0);
	rotate_mat_3.at<double>(1, 0) = rotate_mat.at<double>(1, 0);
	rotate_mat_3.at<double>(0, 1) = rotate_mat.at<double>(0, 1);
	rotate_mat_3.at<double>(1, 1) = rotate_mat.at<double>(1, 1);
	rotate_mat_3.at<double>(0, 2) = rotate_mat.at<double>(0, 2);
	rotate_mat_3.at<double>(1, 2) = rotate_mat.at<double>(1, 2);

	//Получаем комбинированную матрицу линейного преобразования
	Mat combine_transform = rotate_mat_3 * trans_mat_3;
	Mat affine_mat = combine_transform(Rect(0, 0, 3, 2));

	warpAffine(A1, B, affine_mat, B.size());

	//Создаем маску изображения для предотвращения появления темных границ
	//при наложении тайлов друг на друга
	Mat mask = Mat::zeros(A1.size(), 0);
	rectangle(mask, Rect(32, 2, A.cols - 64, A.rows - 4), 255, FILLED);
	//Mat roi = Mat(mask, Rect(36, 2, A1.cols - 72, A1.rows - 4));
	//Point center = Point(roi.cols / 2, roi.rows / 2);
	//ellipse(roi, center, Size(roi.cols*0.55, roi.rows*0.55), 0, 0, 360, 255, FILLED, 0);
	Mat mask1 = Mat::zeros(Size(new_size + d_cols, new_size + d_rows), 0);
	warpAffine(mask, mask1, affine_mat, mask1.size());
	threshold(mask1, mask1, 254, 255, THRESH_BINARY);

	//Применяем полученную маску
	Mat B1 = Mat::zeros(Size(new_size + d_cols, new_size + d_rows), A1.type());
	bitwise_and(B, B, B1, mask1);

	return B1;
}

Point Tiles::giveULpnt()
{
	return ULpnt;
}
int Tiles::giveScale()
{
	return scale;
}
Mat Tiles::giveTile(int x, int y)
{
	return tile[y * cols + x];
}
int Tiles::giveRows()
{
	return rows;
}
int Tiles::giveCols()
{
	return cols;
}

//Определения класса SaveBuffer

SBuffer::SBuffer(int scale)
{
	this->scale = scale;
}
SBuffer::~SBuffer() {}

int SBuffer::addTile(Mat tile, Point pnt, SaveMode mode)
{
	int count = 0;
	if (mode == SAVE)
	{
		Mat buffim = getTile(pnt);
		if (buffim.empty())
		{
			char path[MAXPATHLENGTH];
			memset(path, 0, 64);
			sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, scale, pnt.x, pnt.y);
			buffim = imread(path, IMREAD_UNCHANGED);
		}
		if (!buffim.empty())
		{
			Size r = tile.size();

			Mat gs_tile = Mat::zeros(r, 0);
			Mat mask = Mat::zeros(r, 0);
			Mat mask_inv = Mat::zeros(r, 0);
			Mat tile1 = Mat::zeros(r, buffim.type());
			Mat tile2 = Mat::zeros(r, buffim.type());
			Mat dst_tile = Mat::zeros(r, buffim.type());

			if (tile.channels() == 3)
				cvtColor(tile, gs_tile, COLOR_BGR2GRAY);
			else
				gs_tile = tile;

			threshold(gs_tile, mask, 10, 255, THRESH_BINARY);
			bitwise_not(mask, mask_inv);

			bitwise_and(buffim, buffim, tile1, mask_inv);
			bitwise_and(tile, tile, tile2, mask);

			add(tile1, tile2, dst_tile);

			tileStack[make_pair(pnt.x, pnt.y)] = dst_tile;
		}
		else
		{
			tileStack[make_pair(pnt.x, pnt.y)] = tile;
			count++;
		}
	}
	else
	{
		tileStack[make_pair(pnt.x, pnt.y)] = tile;
		count++;
	}

	return count;
}
Mat SBuffer::getTile(Point pnt)
{
	Mat tile;
	pair <int, int> p = make_pair(pnt.x, pnt.y);
	it = tileStack.find(p);
	if (it != tileStack.end())
		tile = it->second;

	return tile;
}
int SBuffer::write(int min_x, int max_x, int min_y, int max_y)
{
	int count = 0;
	pair <int, int> p;
	char path[MAXPATHLENGTH];
	map <pair <int, int>, Mat>::iterator it2;


	for (it = tileStack.begin(), it2 = it; it != tileStack.end(); it = it2)
	{
		++it2;
		p = it->first;
		int x = p.first;
		int y = p.second;
		if ((x < min_x) or (x > max_x) or (y < min_y) or (y > max_y))
		{
			memset(path, 0, 64);
			int l = sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, scale, p.first, p.second);
			if (imwrite(path, it->second)) count++;
			tileStack.erase(it);
		}
	}
	return count;
}
int SBuffer::writeAll()
{
	int count = 0;
	pair <int, int> p;
	char path[MAXPATHLENGTH];
	map <pair <int, int>, Mat>::iterator it2;

	for (it = tileStack.begin(), it2 = it; it != tileStack.end(); it = it2)
	{
		++it2;
		p = it->first;
		int x = p.first;
		int y = p.second;
		memset(path, 0, 64);
		int l = sprintf_s(path, 64, "%s/%d/%dx%d.png", tiledir, scale, p.first, p.second);
		if (imwrite(path, it->second)) count++;
		tileStack.erase(it);
	}
	return count;
}



