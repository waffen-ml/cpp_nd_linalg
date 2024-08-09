#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cctype>
#include <locale>
#include <unordered_map> 
#include <utility>


inline void trimInplace(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

bool containsOnlyLowerASCIICharacters(std::string& s) {

    for (int i = 0; i < s.length(); i++) {
        int code = int(s[i]);

        if (code < 97 || code > 122)
            return false;

    }

    return true;

}

std::string repeatString(int n, std::string str) {
    std::ostringstream os;
    for (int i = 0; i < n; i++)
        os << str;
    return os.str();
}



int* buildDefaultStride(size_t nDims, int* shape) {
    int m = 1;
    int* stride = new int[nDims];
    stride[nDims - 1] = 1;

    for (int i = nDims - 2; i >= 0; i--)
        stride[i] = stride[i + 1] * shape[i + 1];

    return stride;
}

int calcSize(size_t nDims, int* shape) {
    int m = 1;
    for (int i = 0; i < nDims; i++)
        m *= shape[i];
    return m;
}

bool calcIsDense(size_t nDims, int* shape, int* stride) {
    int m = 1;

    for (int i = nDims - 1; i >= 0; i--) {
        if (stride[i] != m)
            return false;

        m *= shape[i];
    }

    return true;
}




class TensorIterator {
private:
    int start = -1;
    int end = -1;
    int step = 1;

public:
    TensorIterator(int start, int end, int step) {
        if (start < -1 || end < -1 || step == 0 || (end - start) * step < 0)
            throw std::invalid_argument("Invalid arguments");

        this->start = start;
        this->end = end;
        this->step = step;
    }

    TensorIterator(int start, int end) : TensorIterator(start, end, 1) {};

    void compile(int size) {
        if (start >= size || end >= size)
            throw std::invalid_argument("Invalid bounds");
        if (start == -1)
            start = step > 0 ? 0 : size - 1;
        if (end == -1)
            end = step > 0 ? size - 1 : 0;
    }

    int getSize() {
        if (start < 0 || end < 0)
            throw std::exception("Iterator was not compiled yet.");

        int length = abs(start - end) + 1;
        int a = length / abs(step);
        
        if (a * abs(step) < length)
            return a + 1;
        
        return a;
    }

    int getStart() {
        return start;
    }

    int getEnd() {
        return end;
    }

    int getStep() {
        return step;
    }
};

class TensorFilter {
private:
    TensorIterator * iterator = NULL;
    std::vector<int>* indices = NULL;
    int index = -1;

public:
    TensorFilter(int index) {
        this->index = index;
    }
    TensorFilter(std::vector<int> & indices) {
        this->indices = &indices;
    }
    TensorFilter(TensorIterator& iterator) {
        this->iterator = &iterator;
    }

    TensorIterator* getIterator() {
        return iterator;
    }

    std::vector<int>* getIndices() {
        return indices;
    }

    int getIndex() {
        return index;
    }
};


class Tensor {
private:
    int offset = 0;

    float* source;
    size_t sourceSize;

    size_t nDims;
    int* shape;
    int* stride;
    size_t size;
    bool isDense;

    std::string dimToString(int start, int dim) {
        std::stringstream out;

        if (dim + 1 == nDims) {

            out << "[" << std::fixed << std::setprecision(3) << source[start];

            for (int i = 1; i < shape[dim]; i++)
                out << " " << std::fixed << std::setprecision(3) << source[start + i * stride[dim]];

            out << "]";

        }
        else {

            out << "[" << dimToString(start, dim + 1);
            
            for (int i = 1; i < shape[dim]; i++)
                out << std::endl << dimToString(start + i * stride[dim], dim + 1);

            out << "]";
        }

        return out.str();
    }

    int pathToSourcePosition(int path[]) {
        int index = offset;

        for (int i = 0; i < nDims; i++) {
            if (path[i] < 0 || path[i] >= shape[i])
                throw std::invalid_argument("Invalid path");
            index += path[i] * stride[i];
        }

        return index;
    }

    int indexToSourcePosition(int index) {
        if (index < 0 || index >= size)
            throw std::invalid_argument("Invalid path");
        else if (isDense)
            return offset + index;

        int position = offset;
        int w = size;

        for (int i = 0; i < nDims; i++) {
            w /= shape[i];
            position += stride[i] * (index / w);
            index = index % w;
        }
        
        return position;
    }

    void copyDim(Tensor & dest, int destPos, int destDim, int origDim, int origSourcePos, int nFilters, TensorFilter* filters[]) {

        if (origDim >= nDims) {
            dest.set(destPos, source[origSourcePos]);
            return;
        }

        TensorFilter* filter = origDim > nFilters ? NULL : filters[origDim];

        if (filter == NULL) {

            for (int i = 0; i < shape[origDim]; i++)
                copyDim(dest, destPos + dest.getDimStride(destDim) * i,
                    destDim + 1, origDim + 1, origSourcePos + stride[origDim] * i,  nFilters, filters);
            
        }
        else if (filter->getIndices() != NULL) {

            for (int i = 0; i < filter->getIndices()->size(); i++) {

                copyDim(dest, destPos + dest.getDimStride(destDim) * i,
                    destDim + 1, origDim + 1, origSourcePos + stride[origDim] * filter->getIndices()->at(i),
                     nFilters, filters);
            }

        }

        else if (filter->getIndex() != -1) {

            copyDim(dest, destPos, destDim, origDim + 1,
                origSourcePos + filter->getIndex() * stride[origDim], nFilters, filters);

        }

        else if (filter->getIterator() != NULL) {
            TensorIterator* it = filter->getIterator();

            int i = it->getStart();
            int j = 0;

            while ( it->getStep() * (i - it->getEnd()) <= 0) {

                copyDim(dest, destPos + dest.getDimStride(destDim) * j,
                    destDim + 1, origDim + 1,
                    origSourcePos + stride[origDim] * i, nFilters, filters);

                i += it->getStep();
                j++;
            }
        }

        else
            throw std::exception("Invalid filter");

    }


public:

    Tensor(float * source, size_t nDims, int * shape) {
        this->shape = shape;
        this->source = source;
        this->nDims = nDims;
        stride = buildDefaultStride(nDims, shape);
        size = calcSize(nDims, shape);
        sourceSize = size;
        isDense = calcIsDense(nDims, shape, stride);
    }

    Tensor(float* source, size_t nDims, int sourceSize, int offset, int* shape, int* stride) {
        this->source = source;
        this->nDims = nDims;
        this->sourceSize = sourceSize;
        this->offset = offset;
        this->shape = shape;
        this->stride = stride;
        size = calcSize(nDims, shape);
        isDense = calcIsDense(nDims, shape, stride);
    }

    ~Tensor() {
        delete[] stride;
        //delete[] shape;
        //delete[] values;
    }
    
    static Tensor empty(size_t nDims, int* shape) {
        float* values = new float[calcSize(nDims, shape)];
        return Tensor(values, nDims, shape);
    }

    static Tensor zeros(size_t nDims, int* shape) {
        return empty(nDims, shape).fill_(0);
    }

    static Tensor arange(size_t nDims, int* shape, float start, float step) {
        Tensor t = empty(nDims, shape);
        for (int i = 0; i < t.getSize(); i++)
            t.set(i, start + i * step);
        return t;
    }

    static Tensor arange(size_t nDims, int* shape) {
        return arange(nDims, shape, 0, 1);
    }

    std::string toString() {
        return dimToString(offset, 0);
    }

    int getDimSize(int dim) {
        return shape[dim];
    }

    int getDimDefaultStride(int dim) {
        int m = 1;
        for (int i = nDims - 1; i > dim; i--)
            m *= shape[i];
        return m;
    }

    int getDimStride(int dim) {
        return stride[dim];
    }

    int getSize() {
        return size;
    }

    int getNDims() {
        return nDims;
    }

    float get(int i) {
        return source[indexToSourcePosition(i)];
    }

    void set(int i, float val) {
        source[indexToSourcePosition(i)] = val;
    }

    float get(int path[]) {
        return source[pathToSourcePosition(path)];
    }

    void set(int path[], float val) {
        source[pathToSourcePosition(path)] = val;
    }

    Tensor copy() {
        float * newSource = new float[size];
        int* copiedShape = new int[nDims];

        for (int i = 0; i < size; i++)
            newSource[i] = get(i);

        std::copy(shape, shape + nDims, copiedShape);

        return Tensor(newSource, nDims, copiedShape);
    }

    Tensor shallowCopy() {
        int* copiedShape = new int[nDims];
        int* copiedStride = new int[nDims];

        std::copy(shape, shape + nDims, copiedShape);
        std::copy(stride, stride + nDims, copiedStride);

        return Tensor(source, nDims, sourceSize, offset, copiedShape, copiedStride);
    }

    Tensor transpose_() {
        int* newShape = new int[nDims];
        int* newStride = new int[nDims];

        for (int i = 0; i < nDims; i++) {
            newShape[i] = shape[nDims - i - 1];
            newStride[i] = stride[nDims - i - 1];
        }

        return Tensor(source, nDims, sourceSize, offset, newShape, newStride);
    }

    Tensor transpose() {
        return copy().transpose_();
    }

    Tensor T() {
        return transpose();
    }

    Tensor reshape_(size_t newNDims, int* newShape) {
        if (!isDense)
            return copy().reshape_(newNDims, newShape);

        int unknownIdx = -1;
        int unknownDim = 1;
        int m = 1;

        for (int i = 0; i < newNDims; i++) {
            if (newShape[i] < -1 || newShape[i] == 0)
                throw std::invalid_argument("Tensor shape dimensions must be positive or -1 (unknown)");
            else if (newShape[i] == -1 && unknownIdx > -1)
                throw std::invalid_argument("New shape can have only one unknown dimension");
            else if (newShape[i] == -1)
                unknownIdx = i;
            else
                m *= newShape[i];
        }

        if (unknownIdx > -1) {
            unknownDim = size / m;

            if (unknownDim * m != size)
                throw std::invalid_argument("Cannot replace -1 dimension");
        }
        else if (m != size)
            throw std::invalid_argument("Final and current sizes do not match");


        int* newShapeCopy = new int[newNDims];

        std::copy(newShape, newShape + newNDims, newShapeCopy);

        if (unknownIdx > -1)
            newShapeCopy[unknownIdx] = unknownDim;

        return Tensor(source, newNDims, newShapeCopy);
    }

    Tensor reshape(size_t newNDims, int* newShape) {
        return copy().reshape_(newNDims, newShape);
    }

    Tensor selectCopy(size_t querySize, TensorFilter* query[]) {

        if (querySize > nDims)
            throw std::invalid_argument("Max querySize is nDims!");

        std::vector<int> newShape;

        for (int i = 0; i < querySize; i++) {
            TensorFilter* q = query[i];

            if (q == NULL)
                newShape.push_back(shape[i]);
            else if (q->getIndices() != NULL)
                newShape.push_back(q->getIndices()->size());
            else if (q->getIndex() != -1) {
                // do nothing. eat the dimension!
            }
            else if (q->getIterator() != NULL) {
                TensorIterator* it = q->getIterator();
                it->compile(shape[i]);
                newShape.push_back(it->getSize());
            }
            else
                newShape.push_back(shape[i]);
        }

        for (int i = querySize; i < nDims; i++)
            newShape.push_back(shape[i]);

        int* sh = new int[newShape.size()];
        std::copy(newShape.begin(), newShape.end(), sh);

        float* values = new float[calcSize(newShape.size(), sh)];

        Tensor t(values, newShape.size(), sh);

        copyDim(t, 0, 0, 0, offset, querySize, query);

        return t;
    }

    Tensor select(size_t querySize, TensorFilter* query[]) {
        if (querySize > nDims)
            throw std::invalid_argument("Max querySize is nDims!");

        int newOffset = offset;
        std::vector<int> newShape;
        std::vector<int> newStride;

        for (int i = 0; i < querySize; i++) {
            TensorFilter* q = query[i];

            if (q == NULL) {

                newShape.push_back(shape[i]);
                newStride.push_back(stride[i]);

            }
            else if(q->getIndices() != NULL)
                return selectCopy(querySize, query);
            else if (q->getIndex() != -1) {

                newOffset += q->getIndex() * stride[i];

            }
            else if (q->getIterator() != NULL) {

                TensorIterator* it = q->getIterator();
                it->compile(shape[i]);

                if (it->getStep() < 0)
                    return selectCopy(querySize, query);

                newOffset += it->getStart() * stride[i];
                newStride.push_back(stride[i] * it->getStep());
                newShape.push_back(it->getSize());
            }
            else {
                newShape.push_back(shape[i]);
                newStride.push_back(stride[i]);
            }

        }

        for (int i = querySize; i < nDims; i++) {
            newStride.push_back(stride[i]);
            newShape.push_back(shape[i]);
        }

        int* sh = new int[newShape.size()];
        int* st = new int[newShape.size()];

        std::copy(newShape.begin(), newShape.end(), sh);
        std::copy(newStride.begin(), newStride.end(), st);

        return Tensor(
            source, newShape.size(), sourceSize,
            newOffset, sh, st
        );
        
    }

    float sum() {
        float s = 0;

        for (int i = 0; i < size; i++)
            s += get(i);

        return s;
    }

    Tensor add_(float c) {
        for (int i = 0; i < size; i++)
            set(i, get(i) + c);
        return shallowCopy();
    }

    Tensor add(float c) {
        return copy().add_(c);
    }

    Tensor add_(Tensor& w) {
        if (w.getSize() != size)
            throw std::invalid_argument("Tensors do not have equal size");
        for (int i = 0; i < size; i++)
            set(i, get(i) + w.get(i));
        return shallowCopy();
    }

    Tensor add(Tensor& w) {
        return copy().add_(w);
    }

    Tensor mul_(float c) {
        for (int i = 0; i < size; i++)
            set(i, get(i) * c);
        return shallowCopy();
    }

    Tensor mul(float c) {
        return copy().mul_(c);
    }

    Tensor mul_(Tensor& w) {
        if (w.getSize() != size)
            throw std::invalid_argument("Tensors do not have equal size");
        for (int i = 0; i < size; i++)
            set(i, get(i) * w.get(i));
        return shallowCopy();
    }

    Tensor mul(Tensor& w) {
        return copy().mul_(w);
    }

    Tensor fill_(float c) {
        for (int i = 0; i < size; i++)
            set(i, c);
        return shallowCopy();
    }

};





Tensor einsum(std::string formula, Tensor& t1, Tensor * t2) {
    int dIdx = formula.find("->");

    if (dIdx < 0)
        throw std::invalid_argument("Invalid formula: no '->'");

    std::string lhand = formula.substr(0, dIdx);
    std::string rhand = formula.substr(dIdx + 2, formula.length());

    trimInplace(rhand);

    if (!containsOnlyLowerASCIICharacters(rhand))
        throw std::invalid_argument("Invalid formula: rhand side");

    int commaIdx = lhand.find(',');

    if (commaIdx < 0 && t2 != NULL || commaIdx > 0 && t2 == NULL)
        throw std::invalid_argument("Invalid formula: num args");

    std::string firstArgFormula = lhand.substr(0, commaIdx);
    std::string secondArgFormula = t2 == NULL ? "" : lhand.substr(commaIdx + 1, lhand.length());

    trimInplace(firstArgFormula);
    trimInplace(secondArgFormula);

    if (!containsOnlyLowerASCIICharacters(firstArgFormula) || !containsOnlyLowerASCIICharacters(secondArgFormula))
        throw std::invalid_argument("Invalid formula: invalid arg formulas");

    if (firstArgFormula.length() != t1.getNDims() || t2 != NULL && secondArgFormula.length() != t2->getNDims())
        throw std::invalid_argument("Invalid formula: length of formula and dim. did not match");

    // char, (index, size)
    std::unordered_map<char, std::pair<int, int>> m;


    int* firstIndices = new int[firstArgFormula.length()];
    int* secondIndices = t2 == NULL? NULL : new int[secondArgFormula.length()];
    int* resultIndices = new int[rhand.length()];

    
    for (int i = 0; i < firstArgFormula.length(); i++) {
        char c = firstArgFormula[i];
        if (m.find(c) == m.end())
            m[c] = std::pair<int, int>(m.size(), t1.getDimSize(i));
        else if (m[c].second != t1.getDimSize(i))
            throw std::invalid_argument("Invalid formula: incorrect size");
        firstIndices[i] = m[c].first;
    }

    for (int i = 0; i < secondArgFormula.length(); i++) {
        char c = secondArgFormula[i];
        if (m.find(c) == m.end())
            m[c] = std::pair<int, int>(m.size(), t2->getDimSize(i));
        else if (m[c].second != t2->getDimSize(i))
            throw std::invalid_argument("Invalid formula: incorrect size");
        secondIndices[i] = m[c].first;
    }

    for (int i = 0; i < rhand.length(); i++) {
        char c = rhand[i];
        if (m.find(c) == m.end())
            throw std::invalid_argument("Invalid formula: unknown variables in the result");
        resultIndices[i] = m[c].first;
    }

    const int varCount = m.size();
    int* bounds = new int[varCount];
    int* current = new int[varCount];

    for (int i = 0; i < varCount; i++)
        current[i] = 0;

    for (auto x : m) {
        bounds[x.second.first] = x.second.second;
    }

    int* resultShape = new int[rhand.size()];

    for (int i = 0; i < rhand.size(); i++) {
        resultShape[i] = bounds[resultIndices[i]];
    }

    Tensor result = Tensor::zeros(rhand.size(), resultShape);

    int* fpath;
    int* spath;
    int* rpath;

    while (current[0] < bounds[0]) {

        fpath = new int[firstArgFormula.length()];
        spath = t2 == NULL? NULL : new int[secondArgFormula.length()];
        rpath = new int[rhand.length()];

        for (int i = 0; i < firstArgFormula.length(); i++)
            fpath[i] = current[firstIndices[i]];

        for (int i = 0; i < secondArgFormula.length(); i++)
            spath[i] = current[secondIndices[i]];

        for (int i = 0; i < rhand.length(); i++)
            rpath[i] = current[resultIndices[i]];

        float a = t1.get(fpath);
        float b = t2 == NULL? 1 : t2->get(spath);

        result.set(rpath, result.get(rpath) + a * b);

        current[varCount - 1]++;

        for (int i = varCount - 2; i >= 0; i--) {
            if (current[i + 1] >= bounds[i + 1]) {
                current[i + 1] = 0;
                current[i]++;
            }
            else
                break;
        }

        delete[] fpath;
        delete[] spath;
        delete[] rpath;
    }

    delete[] bounds;
    delete[] current;
    delete[] firstIndices;
    delete[] secondIndices;
    delete[] resultIndices;

    return result;
}

int main()
{
        
    Tensor t1 = Tensor::arange(2, new int[2] {4, 4});
    Tensor t2 = Tensor::arange(2, new int[2] {3, 4});

    std::cout << t1.toString() << std::endl;
    std::cout << t2.toString() << std::endl;

    Tensor r = einsum("ii->i", t1, NULL);

    std::cout << r.toString() << std::endl;

}
