#include <stdio.h>
#include <math.h>

double func(double x) {
    return exp(x) - cos(2*x); // Nhap phuong trinh f(x)
}
// CAM DONG VAO DAY
void printSolution(double (*func)(double), double a, double b, int n) {
    printf("%3s %15s %15s %15s %15s\n", "n", "a", "b", "c", "f(c)");
    double c = (a + b) / 2; // initial value
    double z = func(c);
    double delta = fabs((b - a)) / pow(2, n);
    printf("%3d %15.10lf %15.10lf %15.10lf %15.10lf\n", 0, a, b, c,z);
    for (int i = 1; i < n; i++) {
        if (func(c) == 0) {
            printf("Solution: %lf\n", c);
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

    printf("Sai so tuyet doi: %.15e\n", delta);
}
int main() {
    double a = -1; // Nhap khoang cach li a
    double b = -0.1; // Nhap khoang cach li b
    int n = 15; // Nhap so lan lap
    printSolution(func, a, b, n);
    return 0;
}
