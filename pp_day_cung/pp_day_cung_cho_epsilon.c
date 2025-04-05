#include <stdio.h>
#include <math.h>

double f(double x) {
    return 2 * pow(x, 5) + 12 * pow(x, 4) - 5 * pow(x, 3) + 7 * x - 15; // Nhap phuong trinh f(x)
}

// CAM DONG VAO DAY
void printSolution(double (*f)(double), double xk, double d, double m1, double M1, double epsilon) {
    int k = 1;
    double tmp;
    printf("%3s %15s %15s\n", "k", "xk", "|f(xk)|");
    printf("%3d %15.9lf %15.9e\n", 0, xk, fabs(f(xk)));

    tmp = xk;
    xk = xk - ((f(xk) * (xk - d)) / (f(xk) - f(d)));
    printf("%3d %15.9lf %15.9e\n", k, xk, fabs(f(xk)));

    while (fabs(xk - tmp) > (epsilon * m1) / (M1 - m1)) {
        k++;
        tmp = xk;
        xk = xk - ((f(xk) * (xk - d)) / (f(xk) - f(d)));
        printf("%3d %15.9lf %15.15e\n", k, xk, fabs(f(xk)));
    }
}

int main() {
    double d = 3.2; // Nhap diem khoi tao d
    double x0 = 3; // Nhap diem khoi tao x0
    double epsilon = 0.5 * pow(10, -7); // Nhap sai so
    double m1 = 0.466967991; // m1 = min(f'(x))
    double M1 = 0.5150388895; // M1 = max(f'(x))
    printSolution(f, x0, d, m1, M1, epsilon);
    return 0;
}
