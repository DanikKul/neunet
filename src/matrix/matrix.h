#pragma once

#include <cassert>
#include <cmath>
#include <utility>

#define T float

namespace matrix {

    inline T act_sigmoid(T x) {
        return 1.f / (1.f + expf(-x));
    }

    inline T act_linear(T x) {
        return x;
    }

    inline T act_tanhyp(T x) {
        return tanh(x);
    }

    class Matrix {
    public:

        Matrix() = default;

        Matrix(int m_rows, int m_cols, float value = 0) {
            this->rows = m_rows;
            this->cols = m_cols;
            this->matrix = std::vector(m_rows, std::vector(m_cols, value));
        }

        Matrix(int m_rows, int m_cols, std::vector<std::vector<T>> data) {
            this->copyData(std::move(data), m_rows, m_cols);
        }

        Matrix (const Matrix& other) {
            this->copyData(other.matrix, other.rows, other.cols);
        }

        ~Matrix() = default;

        void setCols(int m_cols) {
            this->cols = m_cols;
        }

        void setRows(int m_rows) {
            this->rows = m_rows;
        }

        void setData(std::vector<std::vector<T>> data) {
            this->matrix = std::move(data);
        }

        void copyData(std::vector<std::vector<T>> data, int m_rows, int m_cols) {
            this->rows = m_rows;
            this->cols = m_cols;
            this->matrix = std::vector(m_rows, std::vector<T>(m_cols, 0));
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    this->matrix[i][j] = data[i][j];
                }
            }
        }

        void fillVal(T value) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    this->matrix[i][j] = value;
                }
            }
        }

        void fillRandom(T min, T max, int seed, bool round = false) {
            srand(seed);
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    T value = (T) ((T) rand() / (T) RAND_MAX) * (max - min) + min;
                    if (round) value = (T) std::round(value);
                    this->matrix[i][j] = value;
                }
            }
        }

        void fillRandom(T min, T max, bool round = false) {
            srand(time(nullptr));
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    T value = (T) ((T) rand() / (T) RAND_MAX) * (max - min) + min;
                    if (round) value = (T) std::round(value);
                    this->matrix[i][j] = value;
                }
            }
        }

        void print() const {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    printf("%15f ", matrix[i][j]);
                }
                printf("\n");
            }
        }

        Matrix &operator+=(const Matrix &other) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    matrix[i][j] = this->at(i, j) + other.at(i, j);
                }
            }
            return *this;
        }

        Matrix &operator*=(T value) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    this->set(i, j, this->at(i, j) * value);
                }
            }
            return *this;
        }

        Matrix &operator*=(const Matrix &other) {
            assert(this->cols == other.rows);
            Matrix m(this->rows, other.cols, 0);
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < other.cols; j++) {
                    for (int k = 0; k < this->cols; k++) {
                        m.matrix[i][j] += this->at(i, k) * other.at(k, j);
                    }
                }
            }
            this->copyData(m.matrix, m.rows, m.cols);
            return *this;
        }

        Matrix& operator=(const Matrix& copy) {
            if (&copy != this) {
                this->copyData(copy.matrix, copy.rows, copy.cols);
            }
            return *this;
        }

        Matrix operator-(const Matrix &other) {
            Matrix m(this->rows, this->cols, 0);
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    m.set(i, j, this->at(i, j) - other.at(i, j));
                }
            }
            return m;
        }

        Matrix operator*(const Matrix &other) {
            assert(this->cols == other.rows);
            Matrix m(this->rows, other.cols, 0);
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < other.cols; j++) {
                    for (int k = 0; k < this->cols; k++) {
                        m.matrix[i][j] += this->at(i, k) * other.at(k, j);
                    }
                }
            }
            return m;
        }

        Matrix multiply_like_value(const Matrix &other) const {
            assert(this->cols == other.cols && this->rows == other.rows);
            Matrix m(this->rows, this->cols, 0);
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    m.set(i, j, this->at(i, j) * other.at(i, j));
                }
            }
            return m;
        }

        Matrix& multiply_like_value_inplace(const Matrix &other) {
            assert(this->cols == other.cols && this->rows == other.rows);
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    this->set(i, j, this->at(i, j) * other.at(i, j));
                }
            }
            return *this;
        }

        Matrix operator*(T value) {
            Matrix m(this->rows, this->cols, 0);
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    m.set(i, j, matrix[i][j] * value);
                }
            }
            return m;
        }

        Matrix transpose() {
            Matrix m(this->cols, this->rows, 0);
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    m.set(j, i, matrix[i][j]);
                }
            }
            return m;
        }

        T at(int row, int col) const {
            return matrix[row][col];
        }

        void set(int row, int col, T value) {
            matrix[row][col] = value;
        }

        T sum() const {
            T result = 0;
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    result += matrix[i][j];
                }
            }
            return result;
        }

        Matrix &sigmoid() {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    matrix[i][j] = act_sigmoid(matrix[i][j]);
                }
            }
            return *this;
        }

        Matrix &linear() {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    matrix[i][j] = act_linear(matrix[i][j]);
                }
            }
            return *this;
        }

        Matrix &tanh() {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    matrix[i][j] = act_tanhyp(matrix[i][j]);
                }
            }
            return *this;
        }

        Matrix &square() {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    matrix[i][j] = matrix[i][j] * matrix[i][j];
                }
            }
            return *this;
        }

        int rows;
        int cols;
        std::vector<std::vector<T>> matrix;
    };
}