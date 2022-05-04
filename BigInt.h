#ifndef BIG_INT_H
#define BIG_INT_H

#define CUDA_MEMBER __device__        // Расположение функций класса (device или host)

#include <iostream>
#include <string>

class BigInt {
	std::string value; // значение числа
	bool isNeg; // флаг отрицательности

	CUDA_MEMBER static BigInt karatsuba_mul(const BigInt &a, const BigInt &b); // умножение методом Карацубы

public:
	CUDA_MEMBER BigInt(); // конструктор умолчания (число равно нулю)
	CUDA_MEMBER BigInt(long x); // конструктор преобразования из обычного целого числа
	CUDA_MEMBER BigInt(const std::string &value); // конструктор преобразования из строки
	CUDA_MEMBER BigInt(const BigInt& bigInt); // конструктор копирования

	CUDA_MEMBER const std::string &getValue() const; // получение содержимого строки (строка модуля числа)

	CUDA_MEMBER const bool getIsNeg() const; // получение флага отрицательности числа
	CUDA_MEMBER void setIsNeg(bool isNeg); // установка флага отрицательности числа

	CUDA_MEMBER int sign() const; // получение знака числа
	CUDA_MEMBER const bool isEven() const; // проверка на чётность

	CUDA_MEMBER BigInt abs() const; // получение модуля числа
	CUDA_MEMBER BigInt pow(long n) const; // получение числа в степени n
	CUDA_MEMBER BigInt sqrt(long n = 2) const; // вычисление корня n-ой степени из числа

	// операции сравнения
	CUDA_MEMBER const bool operator==(const BigInt &bigInt) const;
	CUDA_MEMBER const bool operator!=(const BigInt &bigInt) const;

	CUDA_MEMBER const bool operator<(const BigInt &bigInt) const;
	CUDA_MEMBER const bool operator>(const BigInt &bigInt) const;
	CUDA_MEMBER const bool operator<=(const BigInt &bigInt) const;
	CUDA_MEMBER const bool operator>=(const BigInt &bigInt) const;

	// операция присваивания
	CUDA_MEMBER BigInt &operator=(const BigInt &bigInt);

	// унарные плюс и минус
	CUDA_MEMBER BigInt operator+() const &&;
	CUDA_MEMBER BigInt operator-() const &&;

	// арифметические операции
	CUDA_MEMBER BigInt operator+(const BigInt &bigInt) const;
	CUDA_MEMBER BigInt operator-(const BigInt &bigInt) const;
	CUDA_MEMBER BigInt operator*(const BigInt &bigInt) const;
	CUDA_MEMBER BigInt operator/(const BigInt &bigInt) const;
	CUDA_MEMBER BigInt operator%(const BigInt &bigInt) const;
	CUDA_MEMBER BigInt operator<<(size_t n) const;
	CUDA_MEMBER BigInt operator>>(size_t n) const;

	// краткая форма операций
	CUDA_MEMBER BigInt &operator+=(const BigInt &bigInt);
	CUDA_MEMBER BigInt &operator-=(const BigInt &bigInt);
	CUDA_MEMBER BigInt &operator*=(const BigInt &bigInt);
	CUDA_MEMBER BigInt &operator/=(const BigInt &bigInt);
	CUDA_MEMBER BigInt &operator%=(const BigInt &bigInt);
	CUDA_MEMBER BigInt &operator<<=(size_t n);
	CUDA_MEMBER BigInt &operator>>=(size_t n);

	// префиксная форма
	CUDA_MEMBER BigInt &operator++(); // ++v
	CUDA_MEMBER BigInt &operator--(); // --v

	// постфиксная форма
	CUDA_MEMBER BigInt operator++(int); // v++
	CUDA_MEMBER BigInt operator--(int); // v--


	CUDA_MEMBER friend std::ostream &operator<<(std::ostream &stream, const BigInt &bigInt); // вывод числа в выходной поток
	CUDA_MEMBER friend std::istream &operator>>(std::istream &stream, BigInt &bigInt); // ввод числа из входного потока
};

#endif