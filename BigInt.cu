#include "BigInt.h"

#include <iostream>

// конструткор по умолчанию
CUDA_MEMBER BigInt::BigInt() {
	this->isNeg = false;
	this->value = "0";
}

// конструктор из стандартного целого числа
CUDA_MEMBER BigInt::BigInt(long x) {
	this->isNeg = x < 0;
	this->value = std::to_string(isNeg ? -x : x);
}

// конструктор из строки (пустая строка создаст число 0)
CUDA_MEMBER BigInt::BigInt(const std::string &value) {
	if (!value.length()) {
		this->value = "0";
		isNeg = false;

		return;
	}		

	isNeg = value[0] == '-';
	this->value = value.substr(isNeg);

	// определяем число ведущих нулей в строке
	int count = 0;
	while (this->value[count] == '0' && this->value.length() - count > 1)
		count++;

	this->value = this->value.substr(count); // удаляем ведущие нули

	// проверяем "на цифру" каждый символ строки, кидаем исключение, если есть другие символы
	for(size_t i = 0; i < this->value.length(); i++)
		if (this->value[i] < '0' || this->value[i] > '9')
			throw std::string("BigInt(const string &value) - string contain incorrect characters: ") + this->value;
}

// конструктор копирования
CUDA_MEMBER BigInt::BigInt(const BigInt& bigInt) {
	this->value = bigInt.getValue();
	this->isNeg = bigInt.getIsNeg();
}

// получение строки со значением числа
CUDA_MEMBER const std::string &BigInt::getValue() const {
	return value;
}

// получение флага отрицательности числа
CUDA_MEMBER const bool BigInt::getIsNeg() const {
	return isNeg;
}

// изменение флага отрицательности числа
CUDA_MEMBER void BigInt::setIsNeg(bool isNeg) {
	this->isNeg = isNeg;
}

// получение знака числа
CUDA_MEMBER int BigInt::sign() const {
	return (1 - 2 * isNeg) * (value != "0");
}

// проверка на чётность
CUDA_MEMBER const bool BigInt::isEven() const {
	return (value[value.length() - 1] - '0') % 2 == 0;
}

// получение модуля числа
CUDA_MEMBER BigInt BigInt::abs() const {
	return BigInt(value);
}

// получение числа в степени n
CUDA_MEMBER BigInt BigInt::pow(long n) const {
	if (!n)
		return 1;

	if (n & 1)
		return pow(n - 1) * *this;
	else {
		BigInt tmp = pow(n / 2);
		return tmp * tmp;
	}
}

// вычисление корня n-ой степени из числа
CUDA_MEMBER BigInt BigInt::sqrt(long n) const {
	if (n < 2)
		throw std::string("BigInt::sqrt() - n must be 2 or more");

	if (isNeg && ((n & 1)== 0))
		throw std::string("BigInt::sqrt() - value must be positive");

	size_t length = (value.length() + 1) / 2;
	size_t index = 0;

	std::string v(length, '0');

	while (index < length) {
		v[index] = '9';

		while (BigInt(v).pow(n) > this->abs() && v[index] > '0')
			v[index]--;

		index++;
	}

	return isNeg ? std::string("-") + v : v;
}

// проверка на равенство двух чисел (равны, если одного знака и одного значения)
CUDA_MEMBER const bool BigInt::operator==(const BigInt &bigInt) const {
	return (value == bigInt.getValue()) && (isNeg == bigInt.getIsNeg());
}

// проверка на неравенство - отрицание равенства
CUDA_MEMBER const bool BigInt::operator!=(const BigInt &bigInt) const {
	return !(*this == bigInt);
}

// проверка, что число меньше bigInt
CUDA_MEMBER const bool BigInt::operator<(const BigInt &bigInt) const {
	std::string value2 = bigInt.getValue(); // получаем значение второго числа
	size_t len1 = value.length(); // запоминаем длину первого числа
	size_t len2 = value2.length(); // запоминаем длину второго числа

	// если знаки одинаковые, то проверяем значения
	if (isNeg == bigInt.getIsNeg()) {
		// если длины не равны
		if (len1 != len2)
			return (len1 < len2) ^ isNeg; // меньше число с меньшей длинной для положительных и с большей длиной для отрицательных

		size_t i = 0;

		// ищем разряд, в котором значения отличаются
		while (i < len1 && value[i] == value2[i])
			i++;

		// если разряд найден, то меньше число с меньшей цифрой для положительных и с большей цифрой для отрицательных, иначе числа равны
		return (i < len1) && ((value[i] < value2[i]) ^ isNeg);
	}

	return isNeg; // знаки разные, если число отрицательное, то оно меньше, если положительное, то больше
}

// проверка, что число больше bigInt
CUDA_MEMBER const bool BigInt::operator>(const BigInt &bigInt) const {
	return !(*this < bigInt || *this == bigInt);
}

// проверка, что число меньше или равно bigInt
CUDA_MEMBER const bool BigInt::operator<=(const BigInt &bigInt) const {
	return *this < bigInt || *this == bigInt;
}

// проверка, что число больше или равно bigInt
CUDA_MEMBER const bool BigInt::operator>=(const BigInt &bigInt) const {
	return *this > bigInt || *this == bigInt;
}

// операция присваивания
CUDA_MEMBER BigInt &BigInt::operator=(const BigInt &bigInt) {
	value = bigInt.getValue();
	isNeg = bigInt.getIsNeg();

	return *this;
}

// унарный минус, если было отрицательным, возвращаем положительное, иначе отрицательное
CUDA_MEMBER BigInt BigInt::operator-() const && {
	return BigInt(isNeg ? value : std::string("-") + value);
}

// унарный плюс (просто копируем значение числа)
CUDA_MEMBER BigInt BigInt::operator+() const && {
	return BigInt(*this);
}

// умножение методом Карацубы
CUDA_MEMBER BigInt BigInt::karatsuba_mul(const BigInt &a, const BigInt &b) {
    std::string v1 = a.getValue();
    std::string v2 = b.getValue();

    size_t len1 = v1.length();
    size_t len2 = v2.length();
    size_t len = std::max(len1, len2);

    if (len1 + len2 < 9)
        return stol(a.getValue()) * stol(b.getValue());

    len += len % 2;
    size_t n = len / 2;

    BigInt Xr(len1 > n ? v1.substr(len1 - n, n) : v1);
    BigInt Xl(a >> n);
    BigInt Yr(len2 > n ? v2.substr(len2 - n, n) : v2);
    BigInt Yl(b >> n);

    BigInt P1 = karatsuba_mul(Xl, Yl);
    BigInt P2 = karatsuba_mul(Xr, Yr); 
    BigInt P3 = karatsuba_mul(Xl + Xr, Yl + Yr);

    return (P1 << len) + ((P3 - P2 - P1) << n) + P2;
}


// бинарный плюс (сложение двух чисел)
CUDA_MEMBER BigInt BigInt::operator+(const BigInt &bigInt) const {
	bool isAddOp = !(bigInt.getIsNeg() ^ isNeg); // если знаки одинаковые, то выполняем сложение

	if (isAddOp) {
		std::string num2 = bigInt.getValue(); // запоминаем значение второго числа

		size_t len1 = value.length(); // запоминаем длину первого числа
		size_t len2 = num2.length(); // запоминаем длину второго числа
		size_t length = 1 + std::max(len1, len2);  // длина суммы равна максимуму из двух длин + 1 из-за возможного переноса разряда

		char res[length + 1]; // строковый массив для выполнения операции сложения

		res[length - 1] = res[length] = '\0';

		for (size_t i = 0; i < length - 1; i++) {
			int j = length - 1 - i;
			res[j] += ((i < len2) ? (num2[len2 - 1 - i] - '0') : 0) + ((i < len1) ? (value[len1 - 1 - i] - '0') : 0); // выполняем сложение разрядов
	    	res[j - 1] = res[j] / 10; // выполняем перенос в следущий разряд, если он был
		    res[j] = res[j] % 10 + '0'; // оставляем только единицы от возможного переноса и превращаем символ в цифру
		}

		res[0] += '0';
		
		return BigInt(isNeg ? std::string("-") + std::string(res) : std::string(res)); // возвращаем результат, в зависимости от знака`
	}
	else
		return isNeg ? (bigInt - (-BigInt(*this))) : (*this - (-BigInt(bigInt))); // одно из чисел отрицательное, а другое положительное, отправляем на вычитание, меняя знак
}

// бинарный минус (вычитание двух чисел)
CUDA_MEMBER BigInt BigInt::operator-(const BigInt &bigInt) const {
	if (*this == bigInt) 
		return 0; // если числа равны, то какой смысл вычитать?

	// если оба числа положительные, то выполняем вычитание
	if (!isNeg && !bigInt.getIsNeg()) {
		std::string value2 = bigInt.getValue(); // запоминаем значение второго числа
		
		size_t len1 = value.length(); // запоминаем длину первого числа
		size_t len2 = value2.length(); // запоминаем длину второго числа
		size_t length = std::max(len1, len2); // длина результата не превысит максимума длин чисел
		
		bool isNegRes = bigInt > *this; // определяем знак результата

		int a[length], b[length]; // массивы аргументов
		a[0] = b[0] = 0; // обнуляем нулевые элементы массивов

		char res[length + 1]; // строковый массив для результата
		res[length - 1] = res[length] = '\0'; // устанавливаем символ окончания строки

		int sign = (2 * isNegRes - 1); // получаем числовое значение знака результата

		for (size_t i = 0; i < length - 1; i++) {
			a[i] += (i < len1) ? (value[len1 - 1 - i] - '0') : 0; // формируем разряды
			b[i] += (i < len2) ? (value2[len2 - 1 - i] - '0') : 0; // из строк аргументов

			b[i + 1] = -isNegRes; // в зависимости от знака занимаем или не занимаем
			a[i + 1] = isNegRes- 1; // 10 у следующего разряда

			res[length - 1 - i] += 10 + sign * (b[i] - a[i]);
			res[length - 1 - i - 1]  = res[length - 1 - i] / 10;
			res[length - 1 - i] = res[length - 1 - i] % 10 + '0';
		}

		// выполняем операцию с последним разрядом
		a[length - 1] += (length - 1 < len1) * (value[0] - '0');
		b[length - 1] += (length - 1 < len2) * (value2[0] - '0');

		// записываем в строку последний разряд
		res[0] += sign * (b[length - 1] - a[length - 1]) + '0';

		return BigInt(isNegRes ? std::string("-") + std::string(res) : std::string(res)); // возвращаем результат, учитывая знак
	} else // если оба числа отрицательные, то меняем местами, меняем знаки и повторяем вычитание, а если знаки разные, то отправляем на сумму
		return isNeg && bigInt.getIsNeg() ? (-BigInt(bigInt) - (-BigInt(*this))) : (*this + -BigInt(bigInt)); 
}

// бинарная звёздочка (умножение двух чисел)
CUDA_MEMBER BigInt BigInt::operator*(const BigInt &bigInt) const {
	if (value == "0" || bigInt.getValue() == "0")
		return 0; // если один из множителей равен нулю, то результат равен нулю

	std::string value2 = bigInt.getValue(); // запоминаем значение второго числа
		
	size_t len1 = value.length(); // запоминаем длину первого числа
	size_t len2 = value2.length(); // запоминаем длину второго числа
	size_t length = len1 + len2 + 1; // резульат влезет в сумму длин + 1 из-за возможного переноса
	bool isNegRes = isNeg ^ bigInt.getIsNeg(); // флаг отрицательности результата - отрицательный, если числа разных знаков

	if (length < 10) { // умножаем как обычный long
		long res = stol(value) * stol(value2);
		return BigInt(isNegRes ? -res : res);
	}
	else if (length < 2000) { // умножаем в столбик
		int a[length], b[length]; // массивы аргументов и результата

		char res[length + 1]; // строковый массив для результата
		res[length] = '\0'; // устанавливаем символ окончания строки

		// заполняем массивы инверсной записью чисел (с ведущими нулями)
		for (size_t i = 0; i < length; i++) {
			a[i] = (i < len1) ? (value[len1 - 1 - i] - '0') : 0;
			b[i] = (i < len2) ? (value2[len2 - 1 - i] - '0') : 0;
			res[i] = 0;
		}

		// выполняем умножение "в столбик""
		for (size_t i = 0; i < len1; i++) {
		    for (size_t j = 0; j < len2; j++) {
		        res[length - 1 - (i + j)] += a[i] * b[j];
		        res[length - 1 - (i + j + 1)] += res[length - 1 - (i + j)] / 10;
			    res[length - 1 - (i + j)] %= 10;
		    }
		}

		// переписываем результат в строку
		for (size_t i = 0; i < length; i++)
			res[length - 1 - i] += '0';

		return BigInt(isNegRes ? std::string("-") + std::string(res) : std::string(res)); // возвращаем результат, учитывая его знак
	} else { // умножаем по методу Карацубы
		BigInt res = karatsuba_mul(*this, bigInt);
		return isNegRes ? -BigInt(res) : res;
	}
}

// бинарный слеш (деление двух чисел)
CUDA_MEMBER BigInt BigInt::operator/(const BigInt &bigInt) const {
	std::string value1 = value;
	std::string value2 = bigInt.getValue(); // запоминаем значение второго числа

	if (value2[0] == '0')
		throw std::string("Division by zero!"); // нельзя делить на ноль, этому ещё в школе учат

	if (value[0] == '0')
		return 0; // а вот ноль делить можно на всё, кроме нуля, но смысл?

	if (value2 == "1")
		return BigInt(bigInt.getIsNeg() ? -BigInt(*this) : *this); // делить на 1 можно, но смысл?

	size_t zeroes = 0;
	while (value2[value2.length() - 1 - zeroes] == '0')
		zeroes++;

	if (zeroes >= value.length())
		return 0;

	// избавляемся от круглых чисел
	if (zeroes) {
		value1 = value1.substr(0, value1.length() - zeroes);
		value2 = value2.substr(0, value2.length() - zeroes);
	}

	bool isNegRes = isNeg ^ bigInt.getIsNeg(); // считаем знак числа

	BigInt tmp(value2);

	size_t divider_length = value2.length(); // запоминаем длину делителя
	long divider_v = divider_length > 8 ? 0 : atol(value2.c_str()); // если длина больше 8, то обнуляем long'овый делитель, иначе переводим строку в long

	size_t length = value1.length(); // получаем длину делимого
	size_t index = 0; // стартуем с нулевого индекса

	std::string div; // строка результата деления
	std::string v; // строка подчисла (которое делится на делитель в столбик)

	// формируем начальное число для деления
	while (BigInt(v) < tmp && index < length)
		v += value1[index++];

	do {
		int count = 0; // результат деления подчисла на делитель

		// если можем разделить, то делим
		if (BigInt(v) >= tmp) {
			if (divider_length > 8) { // если не входит в long, то делим с помощью вычитания
				BigInt mod = v;

				while (mod >= tmp) {
					mod -= tmp;
					count++;
				}

				v = mod.getValue();
			} else {
				long mod = stol(v);
				count = mod / divider_v;
				v = std::to_string(mod % divider_v);
			}
		}
		
		div += count + '0'; // если не делили, то добавили ноль к результату, иначе добавили результат дедения

		if (index <= length)
			v += value1[index++]; // формируем новое значение для подчисла
	} while (index <= length);

	return BigInt(isNegRes && div != "0" ? std::string("-") + div : div); // возвращаем результат учитывая знак и возможное равенство нулю
}

// бинарный процент (операция взятия остатка от деления) (полностью аналогична делению)
CUDA_MEMBER BigInt BigInt::operator%(const BigInt &bigInt) const {
	std::string value2 = bigInt.getValue();

	if (value2[0] == '0')
		throw std::string("Division by zero!");

	if (value[0] == '0' || value2 == "1")
		return 0;

	if (value.length() < 9 && value2.length() < 9) {
		long res = stol(value) % stol(value2);
		return isNeg ? -res : res;
	}

	BigInt tmp(value2);

	size_t divider_length = value2.length(); // запоминаем длину делителя
	long divider_v = divider_length >= 9 ? 0 : atol(value2.c_str()); // если длина больше 8, то обнуляем long'овый делитель, иначе переводим строку в long

	size_t length = value.length();
	size_t index = 0;
	BigInt mod2 = value;
	std::string v;

	while (BigInt(v) < tmp && index < length)
		v += value[index++];

	do {
		if (BigInt(v) >= tmp) {
			if (divider_v)
				v = std::to_string(stol(v) % divider_v);
			else {
				BigInt mod = v;

				while (mod >= tmp)
					mod -= tmp;

				v = mod.getValue();
			}
		}

		if (index <= length) {
			mod2 = v;
			v += value[index++];				
		}
	} while (index <= length);

	if (mod2.getValue() == "0")
		return 0;

	return isNeg ? -BigInt(mod2) : mod2;
}

CUDA_MEMBER BigInt BigInt::operator<<(size_t n) const {
	return BigInt(std::string(isNeg ? "-" : "") + value + std::string(n, '0'));
}

CUDA_MEMBER BigInt BigInt::operator>>(size_t n) const {
	//if (n >= value.length())
	//	return 0;
	BigInt q = *this;
	std::string qs;
	while(q > 0)
	{
		if (q % 2 == 0)
			qs += "0";
		else
			qs += "1";
		q /= 2;
		// 111101001011111111100000110
	}
	// Invert string
	//std::string qsi;
	qs = qs.substr(1, qs.length() -1 );
	BigInt result(0);
	long int i = 0;
	while (i < qs.length())
	{
		//std::cout << BigInt(2).pow(i) * BigInt(qs[i]) << std::endl;
		//std::cout <<  BigInt(qs[i]) << std::endl;
		if (qs[i] == '1')
			result += BigInt(2).pow(i) * BigInt(1);
		i += 1;
	}
	//std::cout << result << std::endl;
	// for (int i = 0; i < qs.length(); i++)
	// 	qsi += qs[qs.length() - i - 1];
	// //std::cout << qsi  << std::endl;
	// return BigInt(qsi);
	return result;
	//return BigInt(std::string(isNeg ? "-" : "") + value.substr(0, value.length() - n));
}

// краткая запись сложения
CUDA_MEMBER BigInt &BigInt::operator+=(const BigInt &bigInt) {
	return *this = *this + bigInt;
}

// краткая запись вычитания
CUDA_MEMBER BigInt &BigInt::operator-=(const BigInt &bigInt) {
	return *this = *this - bigInt;
}

// краткая запись умножения
CUDA_MEMBER BigInt &BigInt::operator*=(const BigInt &bigInt) {
	return *this = *this * bigInt;
}

// краткая запись деления
CUDA_MEMBER BigInt &BigInt::operator/=(const BigInt &bigInt) {
	return *this = *this / bigInt;
}

// краткая запись взятия остатка
CUDA_MEMBER BigInt &BigInt::operator%=(const BigInt &bigInt) {
	return *this = *this % bigInt;
}

// краткая запись свдига влево
CUDA_MEMBER BigInt &BigInt::operator<<=(size_t n) {
	return *this = *this << n;
}

// краткая запись свдига вправо
CUDA_MEMBER BigInt &BigInt::operator>>=(size_t n) {
	return *this = *this >> n;
}

// префиксная запись инкремента
CUDA_MEMBER BigInt &BigInt::operator++() {
	return *this = *this + 1;
}

// префиксная запись декремента
CUDA_MEMBER BigInt &BigInt::operator--() {
	return *this = *this - 1;
}

// постфиксная запись инкремента
CUDA_MEMBER BigInt BigInt::operator++(int) {
	BigInt res = *this;
	*this = *this + BigInt(1);	

	return res;
}

// постфиксная запись декремента
CUDA_MEMBER BigInt BigInt::operator--(int) {
	BigInt res = *this;
	*this = *this - BigInt(1);

	return res;
}

// вывод числа в выходной поток
CUDA_MEMBER std::ostream &operator<<(std::ostream &stream, const BigInt &bigInt) {
	if (bigInt.getIsNeg())
		stream << "-";

	return stream << bigInt.getValue();
}

CUDA_MEMBER bool BigInt::operator &(int a)
{
	BigInt q = *this;
	std::string qs;
	while(q > 0)
	{
		if (q % 2 == 0)
			qs += "0";
		else
			qs += "1";
		q /= 2;
	}
	if (qs[0] == '1')
		return true;
	else
		return false;
}


// ввод числа из входного потока
CUDA_MEMBER std::istream &operator>>(std::istream &stream, BigInt &bigInt) {
	std::string value;
	stream >> value;
	bigInt = BigInt(value);

	return stream;
}


// CUDA_MEMBER BigInt &operator>>(BigInt &a, int b) {
// 	std::string binary = "";	
// 	int i = 0;
// 	while( i < a.value.length() )
// 	{
// 		switch ( a.value[i] )
// 		{
// 			case '0':
// 				binary += "0000";
// 				break;
// 			case '1':
// 				binary += "0001";
// 				break;
// 			case '2':
// 				binary += "0010";
// 				break;
// 			case '3':
// 				binary += "0011";
// 				break;
	// 		case '4':
	// 			binary += "0100";
	// 			break;
	// 		case '5':
	// 			binary += "0101";
	// 			break;
	// 		case '6':
	// 			binary += "0110";
	// 			break;
	// 		case '7':
	// 			binary += "0111";
	// 			break;
	// 		case '8':
	// 			binary += "1000";
	// 			break;
	// 		case '9':
	// 			binary += "1001";
	// 			break;
	// 	}
	// 	i += 1;
	// }	
	// std::bitset <128> bits(binary);
	// std::cout << bits.to_string() << std::endl;
	// return bits[127];

//}
