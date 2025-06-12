#include <stdio.h>
#include <math.h>
#define e 2.718281828459045

// Can bac ba dung ham cbrt(), can bac hai dung ham sqrt()

double phi(double x) {
    return log(10*x-7); // Nhap ham phi(x)
}

void printSolution(double (*phi)(double), double xk, double q, double epsilon) {
    double tmp = xk;
    int k = 0;
    printf("%3s %15s %15s\n", "k", "xk", "|xk - tmp|");
    if (fabs(phi(xk) - xk) <= epsilon * (1 - q) / q) {
        printf("%3d %15.9lf %15.9lf\n", k, xk, 0.0);
        return;
    }
    do {
        tmp = xk;
        xk = phi(xk);
        k++;
        printf("%3d %15.9lf %15.9e\n", k, xk, fabs(xk - tmp));
    } while (fabs(xk - tmp) > epsilon * (1 - q) / q);
}

int main() {
    double x0 = 3; // Nhap diem khoi tao x0
    // neu phi(x) > 0 thi x0 = a hoac x0 = b
    // neu phi(x) < 0 thi x0 = alpha sao cho a < alpha < (a+b)/2 hoac x0 = beta sao cho (a+b)/2 < beta < b  
    double epsilon = 0.5 * pow(10, -7); // Nhap sai so
    double q = 10.0/23.0; // max|phi'(x)|

    printSolution(phi, x0, q, epsilon);
    return 0;
}