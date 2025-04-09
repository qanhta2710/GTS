#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#define MAX 1000

double find_max(double arr[], int n); 
double find_R(double arr[], int n);
double f(double x, double coeffs[], int n);
double f_prime(double x, double coeffs[], int n);
int count_extrema(double coeffs[], double eps, double R, double extrema[], int n);
void find_intervals(double extrema[], int count, double R, double coeffs[], int n);

int main() {
    int n;
    printf("Nhap bac n cua da thuc: ");
    scanf("%d", &n);

    double *coeffs = (double *)malloc((n+1) * sizeof(double));
    printf("Nhap cac he so cua da thuc: ");
    for (int i = 0; i <= n; i++) {
        scanf("%lf", &coeffs[i]);
    }
    
    double R = find_R(coeffs, n);
    double eps = 1e-7; // Nhập sai số epsilon
    double extrema[MAX];

    int count = count_extrema(coeffs, eps, R, extrema, n);
    find_intervals(extrema, count, R, coeffs, n);
    free(coeffs);
    return 0;
}

double find_max(double arr[], int n) {
    double max_coeff = fabs(arr[0]);
    for (int i = 1; i <= n; i++) {
        if (fabs(arr[i]) > max_coeff) {
            max_coeff = fabs(arr[i]);
        }
    }
    return max_coeff;
}

double find_R(double arr[], int n) {
    double max_coeff = find_max(arr, n);
    double R = 1 + (max_coeff / fabs(arr[0]));
    return R;
}

double f(double x, double coeffs[], int n) {
    double res = 0;
    for (int i = 0; i <= n; i++) {
        res = res * x + coeffs[i];
    }
    return res;
}

double f_prime(double x, double coeffs[], int n) {
    double res = 0;
    for (int i = 0; i < n; i++) {
        res = res * x + coeffs[i] * (n-i);
    }
    return res;
}

int count_extrema(double coeffs[], double eps, double R, double extrema[], int n) {
    double x = -R;
    int count = 0;
    double alpha = 0.001/R; // Trường hợp không tìm được cực trị nào thì hiệu chỉnh hệ số alpha đến khi tìm được
    // Tốt nhất để nguyên đừng ngứa tay động vào lỗi không sửa được đâu
    double prev_fx_prime = f_prime(x, coeffs, n);
    
    if (prev_fx_prime < 0) {
        alpha = -alpha;
    }
    
    while (x <= R) {
        double fx_prime = f_prime(x, coeffs, n);
        double d = x;
        x = x + alpha * fx_prime;
        
        if (prev_fx_prime * fx_prime < 0) {
            alpha = alpha / 2;
            x = d;
            fx_prime = prev_fx_prime;
        }
        else if (fabs(fx_prime) < eps) {
            // Kiểm tra trùng lặp
            int is_duplicate = 0;
            for (int i = 0; i < count; i++) {
                if (fabs(x - extrema[i]) < 0.01) {
                    is_duplicate = 1;
                    break;
                }
            }
            if (!is_duplicate) {
                extrema[count++] = x;
                double last_x = x; 
                alpha = -alpha;   
                
                // Cập nhật x để f'(x) phù hợp với alpha
                x = x + eps;       
                fx_prime = f_prime(x, coeffs, n);
                
                // Kiểm tra dấu f'(x) với alpha
                if ((alpha > 0 && fx_prime < 0) || (alpha < 0 && fx_prime > 0)) {
                    x = last_x;
                    alpha = fabs(alpha); // Đặt lại alpha dương
                    fx_prime = f_prime(x, coeffs, n);
                }
            }
        }
        prev_fx_prime = fx_prime;
    }
    return count;
}

void find_intervals(double extrema[], int count, double R, double coeffs[], int n) {
    if (count == 0) {
        if (f(-R, coeffs, n) * f(R, coeffs, n) < 0) {
            printf("(-%.7f, %.7f)\n", R, R);
        } else {
            printf("Không tìm thấy khoảng cách ly nghiệm.\n");
        }
        return;
    }

    // Sắp xếp các điểm cực trị theo thứ tự tăng dần
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (extrema[i] > extrema[j]) {
                double temp = extrema[i];
                extrema[i] = extrema[j];
                extrema[j] = temp;
            }
        }
    }

    // Kiểm tra khoảng từ -R đến cực trị đầu tiên
    if (f(-R, coeffs, n) * f(extrema[0], coeffs, n) < 0) {
        printf("(-%.7f, %.7f)\n", R, extrema[0]);
    }

    // Kiểm tra các khoảng giữa các cực trị
    for (int i = 0; i < count - 1; i++) {
        if (f(extrema[i], coeffs, n) * f(extrema[i + 1], coeffs, n) < 0) {
            printf("(%.7f, %.7f)\n", extrema[i], extrema[i + 1]);
        }
    }

    // Kiểm tra khoảng từ cực trị cuối cùng đến R
    if (f(extrema[count - 1], coeffs, n) * f(R, coeffs, n) < 0) {
        printf("(%.7f, %.7f)\n", extrema[count - 1], R);
    }
}