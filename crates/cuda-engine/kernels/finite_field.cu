#if FIELD_TYPE == 0
#include "koala_bear.cu"
typedef KoalaBear SmallField;
typedef KoalaBear8 BigField;
typedef KoalaBear8 WhirField;
#else
#error "FIELD must be in [0]"
#endif
