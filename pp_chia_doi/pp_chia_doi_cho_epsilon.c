#include <stdio.h>
#include <math.h>

double func(double x) {
    return 2 * pow(x, 5) + 12 * pow(x, 4) - 5 * pow(x, 3) + 7 * x - 15; // Nhap phuong trinh f(x)
}

void printSolution(double (*func)(double), double a, double b, double epsilon) {
    int n = ceil(log2(fabs(b - a) / epsilon)); 
    printf("So lan lap: %d\n", n);
    printf("%3s %15s %15s %15s %15s\n", "n", "a", "b", "c", "f(c)");
    double c = (a + b) / 2; 
    double z = func(c);
    printf("%3d %15.10lf %15.10lf %15.10lf %15.10e\n", 0, a, b, c, z);
    for (int i = 1; i < n; i++) {
        if (func(c) == 0) {
            printf("Solution: %lf\n", c);
            return;
        } else {
            if (func(a) * z < 0) {
                    b = c;
            } else {
                    a = c;
            }
        }
    c = (a + b) / 2;
    z = func(c);
    printf("%3d %15.10lf %15.10lf %15.10lf %15.10e\n", i, a, b, c, z);
    }
}
int main() {
    // Nhap khoang cach li (a, b)
    double a = 0; 
    double b = 1; 
    // Nhap sai so
    double epsilon = 0.5 * pow(10, -7); 
    printSolution(func, a, b, epsilon);
    return 0;
}
