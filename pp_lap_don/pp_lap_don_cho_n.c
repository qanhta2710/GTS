#include <stdio.h>
#include <math.h>

double phi(double x) {
    return 1 / (x + 3); // Nhap ham phi(x)
}

// CAM DONG VAO DAY
void printSolution(double (*phi)(double), double xk, int n, double q) {
    double tmp;
    printf("%3s %15s\n", "k", "xk");

    printf("%3d %15.9lf\n", 0, xk);

    for (int k = 1; k <= n; k++) {
        tmp = xk;
        xk = phi(xk);
        printf("%3d %15.9lf\n", k, xk);
    }
    printf("Sai so tuyet doi: %15.15e\n", q / (1 - q) * fabs(xk - tmp));
}

int main() {
    double x0 = 0; // Nhap diem khoi tao x0
    int n = 5; // Nhap so lan lap
    double q = 1.0/9.0; // max|phi'(x)|
    printSolution(phi, x0, n, q);
    return 0;
}