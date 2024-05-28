package com.pets.pets_service.Controllers;

import com.pets.pets_service.Exception.ResourceNotFoundException;
import com.pets.pets_service.Models.Pet;
import com.pets.pets_service.Repositories.ClientRepo;
import com.pets.pets_service.Repositories.PetRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@CrossOrigin(origins = "http://localhost:4200")
@RestController
@RequestMapping("api/v1")
public class PetController {
    @Autowired
    private PetRepo petRepo ;
    @Autowired
    private ClientRepo clientRepo ;

    @GetMapping("/pets")
    public List<Pet> getAllPets(){
        return petRepo.findAll();
    }
    @GetMapping("/pets/{id}")
    public ResponseEntity<Pet> getPetById(@PathVariable(value = "id") Integer petId)
            throws ResourceNotFoundException {

        Pet pet = petRepo.findById(petId)
                .orElseThrow(() -> new ResourceNotFoundException("Pet not found"));
        return ResponseEntity.ok().body(pet);
    }
    @PostMapping("/{clientId}/pets")
    public Pet createAccount(@PathVariable(value = "clientId") Integer clientID,@Valid @RequestBody Pet petRequest) {
        Pet pet= clientRepo.findById(clientID).map(client->{
            petRequest.setClient(client);
            return petRepo.save(petRequest);
        }).orElseThrow(() -> new ResourceNotFoundException("Client not found"));
        return pet;
    }

    @PutMapping("/pets/{id}")
    public ResponseEntity<Pet> updatePet(@PathVariable(value = "id") Integer petId,
                                                 @Valid @RequestBody Pet petDetails) throws ResourceNotFoundException {
        Pet pet = petRepo.findById(petId)
                .orElseThrow(() -> new ResourceNotFoundException("Pet not found"));
        pet.setSex(petDetails.getSex());
        pet.setAge(petDetails.getAge());
        pet.setFurColor(petDetails.getFurColor());
        pet.setForAdoption(pet.isForAdoption());
        pet.setImgUrl(petDetails.getImgUrl());
        pet.setPetType(petDetails.getPetType());
        pet.setClient(petDetails.getClient());
        final Pet updatedPet = petRepo.save(pet);
        return ResponseEntity.ok(updatedPet);
    }
    @DeleteMapping("/pets/{id}")
    public Map<String, Boolean> deleteAccount(@PathVariable(value = "id") Integer petId)
            throws ResourceNotFoundException {
        Pet pet = petRepo.findById(petId)
                .orElseThrow(() -> new ResourceNotFoundException("Pet not found"));
        petRepo.delete(pet);
        Map<String, Boolean> response = new HashMap<>();
        response.put("deleted", Boolean.TRUE);
        return response;
    }
}
