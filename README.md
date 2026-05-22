# Proyecto 6: Computer Vision

*Escribe aquí una breve descripción de tu proyecto*
#Voy a escoger el proyecto 1:el sistema de asistencia automatizada con #reconocimiento facial: Imagina un salón de clases o una oficina donde #la asistencia se toma automáticamente mediante un sistema de #reconocimiento facial. Este proyecto consiste en desarrollar un #sistema que detecte y registre automáticamente la asistencia de las #personas al reconocer sus rostros en tiempo real a través de una #cámara. Esto no solo facilita la gestión del tiempo para maestros y #gerentes, sino que también reduce la posibilidad de errores humanos y #asegura que los registros de asistencia sean precisos y estén #actualizados. Además, podría extenderse para notificar a los padres o #supervisores cuando alguien no asista, mejorando la comunicación y la #gestión del personal o estudiantes.

*Escribe un instructivo de cómo podemos utilizar tu software (incluye instrucciones para crear entorno virtual)*
## Uso del sistema de asistencia facial

1. Crear un entorno virtual:

```bash
#python -m venv .venv
#source .venv/bin/activate  
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Crear la estructura de datos:

```text
dataset/teachers/<nombre>.jpg
 dataset/students/<nombre>.jpg
```

4. Ejecutar el sistema:

```bash
python attendance_deepface.py
```

5. Presionar `q` para cerrar la cámara.

El reporte de asistencia se almacenará en `attendance.csv`.
