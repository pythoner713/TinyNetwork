#include<bits/stdc++.h>
#define _N_ N + 5
#define _L1_ L1 + 5
#define _L2_ L2 + 5
#pragma gcc optimize(2)
using namespace std;

int ans, res, cnt;
const double r = 10;
const int N = 28, L1 = 16, L2 = 16, size = 100;

double a[_N_][_N_], l1[_L1_], l2[_L2_], z[10], m1[_L1_], m2[_L2_], m3[10];
double w1[_L1_][_N_][_N_], w2[_L2_][_L1_], w3[10][_L2_];
double b1[_L1_], b2[_L2_], b3[10];

double dl1[_L1_], dl2[_L2_];
double dw1[_L1_][_N_][_N_], dw2[_L2_][_L1_], dw3[10][_L2_];
double db1[_L1_], db2[_L2_], db3[10];

inline double s(double x){
	return 1 / (1 + exp(-x));
}

inline double s_(double x){
	return s(x) * (1 - s(x));
}

void read(int x){
	char FILE[42] = "";
	sprintf(FILE, "data/images/%d.txt", x);
	freopen(FILE, "r", stdin);
	for(int i = 1; i <= N; i++)
		for(int j = 1; j <= N; j++)
			cin >> a[i][j];
	cin >> ans;
	cin.clear();
	fclose(stdin);
}

void print(){
	string pix = " .*#";
	for(int i = 1; i <= N; i++){
		for(int j = 1; j <= N; j++){
			putchar(pix[a[i][j] / 64]);
		}
		putchar('\n');
	}
	printf("DIGIT: %d\n", ans);
} 

void Forward_Propagation(){
	memset(l1, 0, sizeof(l1));
	memset(l2, 0, sizeof(l2));
	memset(z, 0, sizeof(z));
	for(int now = 1; now <= L1; now++){
		for(int i = 1; i <= N; i++){
			for(int j = 1; j <= N; j++){
				l1[now] += a[i][j] * w1[now][i][j];
			}
		}
		m1[now] = l1[now] + b1[now];
		l1[now] = s(m1[now]);
	}
	for(int now = 1; now <= L2; now++){
		for(int i = 1; i <= L1; i++){
			l2[now] += l1[i] * w2[now][i];
		}
		m2[now] = l2[now] + b2[now];
		l2[now] = s(m2[now]);
	}
	for(int now = 0; now <= 9; now++){
		for(int i = 1; i <= L2; i++){
			z[now] += l2[i] * w3[now][i];
		}
		m3[now] = z[now] + b3[now];
		z[now] = s(m3[now]);
	}
	
	double maxx = 0;
	for(int i = 0; i <= 9; i++){
		if(maxx < z[i]){
			maxx = z[i];
			res = i;
		}
		printf("%d: %.6lf\n", i, z[i]);
	}
	printf("RESULT: %d\n", res);
	cnt += (res == ans);
}

void Back_Propagation(){
	double cost = 0;
	for(int i = 0; i <= 9; i++){
		double t = (i == ans);
		cost += (z[i] - t) * (z[i] - t);
	}
	printf("COST: %.6lf\n", cost);
	
	// LAST LAYER
	for(int i = 0; i <= 9; i++){
		double t = (i == ans);
		for(int j = 1; j <= L2; j++){
			dw3[i][j] += l2[j] * s_(m3[i]) * 2 * (z[i] - t);
			dl2[j] += w3[i][j] * s_(m3[i]) * 2 * (z[i] - t);
		}
		db3[i] += s_(m3[i]) * 2 * (z[i] - t);
	}
	// SECOND LAYER
	for(int i = 1; i <= L2; i++){
		for(int j = 1; j <= L1; j++){
			dw2[i][j] += l1[j] * s_(m2[i]) * dl2[i];
			dl1[j] += w2[i][j] * s_(m2[i]) * dl2[i];
		}
		db2[i] += s_(m2[i]) * dl2[i];
	}
	// FIRST LAYER
	for(int i = 1; i <= L1; i++){
		for(int j = 1; j <= N; j++){
			for(int k = 1; k <= N; k++){
				dw1[i][j][k] += a[j][k] * s_(m1[i]) * dl1[i];
			}
		}
		db1[i] += s_(m1[i]) * dl1[i];
	}
}

void update(){
	double R = r / size;
	for(int i = 1; i <= L1; i++){
		for(int j = 1; j <= N; j++){
			for(int k = 1; k <= N; k++){
				w1[i][j][k] -= dw1[i][j][k] * R;
			}
		}
		b1[i] -= db1[i] * R;
	}
	for(int i = 1; i <= L2; i++){
		for(int j = 1; j <= L1; j++){
			w2[i][j] -= dw2[i][j] * R;
		}
		b2[i] -= db2[i] * R;
	}
	for(int i = 0; i <= 9; i++){
		for(int j = 1; j <= L2; j++){
			w3[i][j] -= dw3[i][j] * R;
		}
		b3[i] -= db3[i] * R;
	}
	
	memset(dl1, 0, sizeof(dl1));
	memset(dl2, 0, sizeof(dl2));
	memset(dw1, 0, sizeof(dw1));
	memset(dw2, 0, sizeof(dw2));
	memset(dw3, 0, sizeof(dw3));
	memset(db1, 0, sizeof(db1));
	memset(db2, 0, sizeof(db2));
	memset(db3, 0, sizeof(db3));
}

int main(){
	freopen("log.txt", "w", stdout); 
	for(int i = 1; i <= 5000; i++){
		read(i);
		print();
		Forward_Propagation();
		Back_Propagation();
		if(i % size == 0) update();
	}
	printf("CORRECT: %d\n", cnt);
	return 0;
}
