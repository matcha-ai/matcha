# Dtype promotion rules

When an operation expecting multiple input tensors of the same `Dtype`, 
(such as [elementwise binary](tensor/operations/elementwise-binary)
operations) receives inputs of different dtypes, implicit type casting
to a target dtype is performed based on the following rules:


|       |Bool   |Byte   |Ushort |Uint   |Ulong  |Sbyte  |Short  |Int    |Long   |Half   |Float  |Double |Cuint  |Cint   |Cfloat |Cdouble|
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|**Bool**   |Bool   |Byte   |Ushort |Uint   |Ulong  |Sbyte  |Short  |Int    |Long   |-      |Float  |Double |Cuint  |Cint   |Cfloat |Cdouble|
|**Byte**   |       |Byte   |Ushort |Uint   |Ulong  |Short  |Short  |Int    |Long   |-      |Float  |Double |Cuint  |Cint   |Cfloat |Cdouble|
|**Ushort** |       |       |Ushort |Uint   |Ulong  |Int    |Int    |Int    |Long   |-      |Float  |Double |Cuint  |Cint   |Cfloat |Cdouble|
|**Uint**   |       |       |       |Uint   |Ulong  |Long   |Long   |Long   |Long   |-      |Float  |Double |Cuint  |Cint   |Cfloat |Cdouble|
|**Ulong**  |       |       |       |       |Ulong  |Long   |Long   |Long   |Long   |-      |Float  |Double |Cuint  |Cint   |Cfloat |Cdouble|
|**Sbyte**  |       |       |       |       |       |Sbyte  |Short  |Int    |Long   |-      |Float  |Double |Cint   |Cint   |Cfloat |Cdouble|
|**Short**  |       |       |       |       |       |       |Short  |Int    |Long   |-      |Float  |Double |Cint   |Cint   |Cfloat |Cdouble|
|**Int**    |       |       |       |       |       |       |       |Int    |Long   |-      |Float  |Double |Cint   |Cint   |Cfloat |Cdouble|
|**Long**   |       |       |       |       |       |       |       |       |Long   |-      |Double |Double |Cint   |Cint   |Cdouble|Cdouble|
|**Half**   |       |       |       |       |       |       |       |       |       |-      |-      |-      |-      |-      |-      |-      |
|**Float**  |       |       |       |       |       |       |       |       |       |       |Float  |Double |Cfloat |Cfloat |Cfloat |Cdouble|
|**Double** |       |       |       |       |       |       |       |       |       |       |       |Double |Cdouble|Cdouble|Cdouble|Cdouble|
|**Cuint**  |       |       |       |       |       |       |       |       |       |       |       |       |Cuint  |Cint   |Cfloat |Cdouble|
|**Cint**   |       |       |       |       |       |       |       |       |       |       |       |       |       |Cint   |Cfloat |Cdouble|
|**Cfloat** |       |       |       |       |       |       |       |       |       |       |       |       |       |       |Cfloat |Cdouble|
|**Cdouble**|       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |Cdouble|
