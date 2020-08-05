#pragma once

const char basedir[] = "./Database/BottomPhoto/photo"; //���� � ����
const char datadir[] = "./Database/BottomPhoto/data.txt"; //���� � ������ �� ����
const char tiledir[] = "./Database/BottomPhoto/Tiles"; //���� � ���� ������
const char logdir[] = "./logs.csv"; //���� � �����

const int MAXPATHLENGTH = 64;
const int MaxScale = 19; //������������ ����� ������� �����
const int patch_size = 256; //������ �����
const double max_half_width = 33554432; //�������� ������������ ������ ������� �����, ���������� � ��������

const float focus = 0.008; //�������� ���������� ������
const int px = 90; //���������� �������� �� ��������� ������� ������
const float Hnom = 1.6; //����������� ������ ������ 
const float mashtab = Hnom / focus; //������� ������ 1:m �� ����������� ������ Hnom
const float PX = px * 1000 / mashtab; //���������� �������� �� ���� � �������� �����������, ���������� �� ������ Hnom

const float gamma = 0.6; //����������� �����-���������
const float claheLimit = 3.5; //������� ������������� ��� ������������� ������� � ������������� �������

enum SaveMode { SAVE, REPLACE };